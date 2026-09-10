import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
import matplotlib.pyplot as plt


from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
from lib.test.analysis.extract_results import calc_err_center, calc_iou_overlap, calc_seq_err_robust
import torch


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False, ground_truth = None):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_
        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)
        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"
        gt_boxes = []
        with open(ground_truth, "r") as f:
            for frame_id, line in enumerate(f):
                gt_boxes.append(eval(line.strip()))

        output_boxes = []
        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        # cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        optional_box = [gt_boxes[0][0], gt_boxes[0][1], gt_boxes[0][2], gt_boxes[0][3]] 
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                # cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                        #    1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break
        frame_id = 0
        plt.ion()

        fig, ax = plt.subplots(figsize=(6,4))
        iou_history = []

        line, = ax.plot([], [], 'b-')

        ax.set_xlabel("Frame")
        ax.set_ylabel("IoU")
        ax.set_ylim(0, 1)
        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)
            
            # cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
            #              (0, 255, 0), 2)

            font_color = (0, 0, 0)
            # cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            font_color, 1)
            # cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            font_color, 1)
            # cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #            font_color, 1)

            # gt = [int(v) for v in gt_boxes[frame_id]]
            # pred_bb = torch.tensor([state], dtype=torch.float32)
            # anno_bb = torch.tensor([gt], dtype=torch.float32)
            # iou = calc_iou_overlap(pred_bb, anno_bb)[0].item()
            # iou_history.append(iou)
            # if frame_id % 5 == 0:      # cập nhật mỗi 5 frame
            #     line.set_xdata(range(len(iou_history)))
            #     line.set_ydata(iou_history)

            #     ax.set_xlim(0, max(50, len(iou_history)))

            #     fig.canvas.draw()
            #     fig.canvas.flush_events()
            # cv.putText(
            #     frame_disp,
            #     f"IoU: {iou:.3f}",
            #     (20, 110),
            #     cv.FONT_HERSHEY_COMPLEX_SMALL,
            #     1,
            #     (255, 0, 0),
            #     1
            # )
            # cv.rectangle(
            #     frame_disp,
            #     (gt[0], gt[1]),
            #     (gt[0] + gt[2], gt[1] + gt[3]),
            #     (0, 0, 255),
            #     2
            # )


            # # Display the resulting frame
            # cv.putText(frame_disp, "Pred", (state[0], state[1]-5),
            # cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # cv.putText(frame_disp, "GT", (gt[0], gt[1]-5),
            #         cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # # Hiển thị
            # cv.imshow(display_name, frame_disp)

            frame_id += 1
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break
            # elif key == ord('r'):
            #     frame_id = 0
            #     ret, frame = cap.read()
            #     frame_disp = frame.copy()

            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
            #                (0, 0, 0), 1)

            #     cv.imshow(display_name, frame_disp)
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        # cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')
        # seq_length = len(gt_boxes)
        # plot_bin_gap = 0.05
        # pred_bb = torch.tensor(output_boxes)
        # anno_bb = torch.tensor(gt_boxes)
        # err_center = calc_err_center(pred_bb, anno_bb)
        # err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
        # err_overlap = calc_iou_overlap(pred_bb, anno_bb)
        # threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
        # threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
        # threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0
        # ave_success_rate_plot_overlap = (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / seq_length
        # ave_success_rate_plot_center = (err_center.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / seq_length
        # ave_success_rate_plot_center_norm = (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / seq_length


        # plt.figure(figsize=(8,5))
        # plt.plot(
        #     threshold_set_center_norm.numpy(),
        #     ave_success_rate_plot_center_norm.numpy(),
        #     linewidth=2
        # )

        # plt.xlabel("Threshold")
        # plt.ylabel("Precision")
        # plt.title("Precision Nprm plot")
        # plt.grid(True)
        # plt.savefig("success_plot.png")
        # print(1)
    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



