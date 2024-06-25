from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec


def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)
    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(50000)
    window_size = FieldReference(default=1)
    action_dim = FieldReference(7)

    config = dict(
        pretrained_path="/shared/projects/mirage2/octo_base_finetune_augmented/octo_base_finetune_augmented/experiment_20240605_155013",
        pretrained_step=25000,
        batch_size=400,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir="/shared/projects/mirage2/octo_base_finetune_augmented_25_backup_in_domain",
        seed=42,
        wandb=dict(
            project="octo_base_finetune_augmented", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix="in_domain",
                data_dir="/shared/projects/mirage2/in_domain_with_language",
                load_camera_views=("primary", "wrist"),
                load_depth=False,
                force_recompute_dataset_statistics=False,
            ),
            traj_transform_kwargs=dict(
                action_horizon=4,
                max_action_dim=action_dim,
                task_augment_strategy="delete_and_rephrase",
                task_augment_kwargs=dict(
                    paraphrases_repo="rail-berkeley/OXE_paraphrases",
                    paraphrases_filename="paraphrases_oxe.pkl",
                    rephrase_prob=0.5,
                ),
            ),
            batch_size=400,
            shuffle_buffer_size=10000,
            balance_weights=True,
            # name="test",
            # data_dir="/shared/projects/mirage2/octo_oxe",
        ),
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,  # if you are using grad accumulation, you need to adjust max_steps accordingly
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=400,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
        eval_datasets=["franka_tiger", "franka_cloth", "jaco_bowl", "jaco_cup"],
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
    )
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    frame_transform_kwargs = dict(
        resize_size={
            "primary": (256, 256),  # workspace (3rd person) camera is at 256x256
            "wrist": (128, 128),  # wrist camera is at 128x128
        },
        image_augment_kwargs=dict(
            primary=workspace_augment_kwargs,
            wrist=wrist_augment_kwargs,
        ),
    )
    # If the default data loading speed is too slow, try these:
    config[
        "frame_transform_threads"
    ] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["dataset_kwargs"]["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    config["dataset_kwargs"]["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
