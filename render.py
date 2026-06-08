import os
import torch
import sys
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.mesh_utils import GaussianExtractor
from utils.render_utils import generate_path, create_videos
from utils.general_utils import safe_state


def render_sets(dataset, opt, pipe, checkpoint, gaussian_dim, time_duration, rot_4d, force_sh_3d,
                num_pts, num_pts_ratio, render_train, render_test, render_traj, n_frames):

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(
        dataset.sh_degree,
        gaussian_dim=gaussian_dim,
        time_duration=time_duration,
        rot_4d=rot_4d,
        force_sh_3d=force_sh_3d,
        sh_degree_t=2 if pipe.eval_shfs_4d else 0
    )

    assert checkpoint, "No checkpoint provided for rendering"
    scene = Scene(dataset, gaussians, shuffle=False, num_pts=num_pts,
                  num_pts_ratio=num_pts_ratio, time_duration=time_duration)

    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, None)

    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    render_base = os.path.join(dataset.model_path, "renders", "ours_{}".format(first_iter))

    if render_train:
        train_dir = os.path.join(render_base, "train")
        os.makedirs(train_dir, exist_ok=True)
        print("Rendering training cameras ...")
        gaussExtractor.reconstruction(scene.getTrainCameras(), train_dir, stage="validation")
        gaussExtractor.export_image(train_dir, mode="validation")
        print(f"Train renders saved to {train_dir}")

    if render_test:
        test_dir = os.path.join(render_base, "test")
        os.makedirs(test_dir, exist_ok=True)
        print("Rendering test cameras ...")
        gaussExtractor.reconstruction(scene.getTestCameras(), test_dir, stage="validation")
        gaussExtractor.export_image(test_dir, mode="validation")
        print(f"Test renders saved to {test_dir}")

    if render_traj:
        traj_dir = os.path.join(render_base, "traj")
        os.makedirs(traj_dir, exist_ok=True)
        print(f"Rendering trajectory ({n_frames} frames) ...")
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_frames)
        gaussExtractor.reconstruction(cam_traj, traj_dir, stage="trajectory")
        gaussExtractor.export_image(traj_dir, mode="trajectory")
        create_videos(
            base_dir=traj_dir,
            input_dir=traj_dir,
            out_name="render_traj",
            num_frames=n_frames
        )
        print(f"Trajectory renders and video saved to {traj_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--start_checkpoint", type=str, default=None,
                        help="Path to checkpoint .pth file. If omitted, auto-detected from model_path.")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")

    parser.add_argument("--render_train", action="store_true", help="Render training cameras")
    parser.add_argument("--render_test", action="store_true", default=True, help="Render test cameras")
    parser.add_argument("--render_traj", action="store_true", help="Render interpolated camera trajectory")
    parser.add_argument("--n_frames", type=int, default=480, help="Number of trajectory frames")

    args = parser.parse_args(sys.argv[1:])

    cfg = OmegaConf.load(args.config)

    def recursive_merge(key, host):
        if isinstance(host[key], DictConfig):
            for key1 in host[key].keys():
                recursive_merge(key1, host[key])
        else:
            assert hasattr(args, key), key
            setattr(args, key, host[key])

    for k in cfg.keys():
        recursive_merge(k, cfg)

    # Auto-detect checkpoint from model_path if not provided
    if not args.start_checkpoint:
        model_path = args.model_path
        best = os.path.join(model_path, "chkpnt_best.pth")
        if os.path.exists(best):
            args.start_checkpoint = best
        else:
            candidates = sorted(
                [f for f in os.listdir(model_path) if f.startswith("chkpnt") and f.endswith(".pth")],
                key=lambda f: int("".join(filter(str.isdigit, f)) or 0)
            )
            assert candidates, f"No checkpoint found in {model_path}"
            args.start_checkpoint = os.path.join(model_path, candidates[-1])
        print(f"Auto-detected checkpoint: {args.start_checkpoint}")

    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    render_sets(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.start_checkpoint,
        args.gaussian_dim, args.time_duration,
        args.rot_4d, args.force_sh_3d,
        args.num_pts, args.num_pts_ratio,
        args.render_train, args.render_test, args.render_traj, args.n_frames
    )

    print("\nComplete.")
