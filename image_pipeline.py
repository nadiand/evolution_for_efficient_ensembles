from typing import Callable

import numpy as np
import logging
import transformations as trans


class ImagePipeline:
    def __init__(
        self,
        transformations: list[Callable],
        parameters: list[int],
        use_both_lighting=True,
        augment_mask=True,
        enforce_shape=False,
    ):
        """The image transform pipeline as preprocessing to the segmentation model

        Args:
            transformations (list): The transform method names
            parameters (list[int]): A list of parameters to the methods
            use_both_lighting (bool, optional): Use both ring and coaxial light. Defaults to True.
            augment_mask (bool, optional): Augment the mask. Defaults to True.
            enforce_shape (bool, optional): Enforce consistent shape before and after transform.
                Enforced the same numpy shapes. Defaults to False.
        """
        self.n_pipelines = len(transformations)
        self.n_transformations = len(transformations[0])

        self.parameters = parameters
        self.augment_mask = augment_mask
        self.enforce_shape = enforce_shape
        self.transformations = transformations
        self.use_both_lighting = use_both_lighting

    def __call__(self, img: np.ndarray, mask: np.ndarray = None, transform_idx: int = None):

        pipelines = []
        # iterate over the 1st pipeline
        for i in range(self.n_transformations):
            # Generate a tuple for each transform
            parameters = [param[i] for param in self.parameters]
            transform = [transform[i] for transform in self.transformations]
            pipelines.append((parameters, transform))

        # Optional to specify the transformations, to single out transformations
        if transform_idx is not None:
            pipelines = [pipelines[transform_idx]]

        for params, func in pipelines:
            img_shape = img.shape
#            print(img_shape, func)

#            logging.info(func)
            img = func[0](img, **params[0], is_mask=False)
#            if self.use_both_lighting:
                # TODO: Untested for batching
                # Expand dims to (B, H, W, 1) for both lighting conditions
#                img0 = np.expand_dims(img[:, :, :, 0], axis=[-1])
#                img1 = np.expand_dims(img[:, :, :, 1], axis=[-1])
#                print('expand', img0.shape)

                # Transforms the image according to the current transform
#                img0 = func[0](img0, **params[0], is_mask=False)
#                img1 = func[1](img1, **params[1], is_mask=False)
                # Outputs (B, H, W, 1)
#                print('trasnform', img0.shape)

                # Concatenates the image by the last dimension
#                img = np.concatenate([img0, img1], axis=-1)
                # Outputs (B, H, W, 2)
#                print('concat', img.shape)
#            else:
                # if both lightings are used but only a single pipeline
                # is used.
#                n_images = img.shape[-1]
#                if n_images == 1:
#                    img = func[0](img, **params, is_mask=False)
#                else:
#                    img0 = np.expand_dims(img[:, :, :, 0], axis=[-1])
#                    img1 = np.expand_dims(img[:, :, :, 1], axis=[-1])

#                    img0 = func[0](img0, **params, is_mask=False)
#                    img1 = func[0](img1, **params, is_mask=False)

                    # Concatenates the image by the last dimension
#                    img = np.concatenate([img0, img1], axis=-1)

#            print(img_shape, img.shape, 'not same bitcxh')
            # The image shape should remain the same
            if img_shape != img.shape and self.enforce_shape:
                raise ValueError(
                    f"The augmentation {func[0].__name__} changed the image shape: "
                    f"original shape: {img_shape}, new shape: {img.shape}"
                )
            # Apply to the label, if the label exists, attribute is specified, and
            # configured to do so.
            if (
                mask is not None
                and getattr(func[0], "apply_to_mask")
                and self.augment_mask
            ):
                # TODO: Currently, only applicable to resize_by_factor augmentation

                mask_shape = mask.shape
                mask = func[0](mask, **params[0], is_mask=True)

                # The mask shape should remain the same
                if mask_shape != mask.shape and self.enforce_shape:
                    raise ValueError(
                        f"The augmentation {func.__name__} changed the mask shape: "
                        f"original shape: {mask_shape}, new shape: {mask.shape}"
                    )
        return img, mask

    def show_transformations(self, img: np.ndarray):
        pipelines = []

        # iterate over the 1st pipeline
        for i in range(self.n_transformations):
            # Generate a tuple for each transform
            parameters = [param[i] for param in self.parameters]
            transform = [transform[i] for transform in self.transformations]
            pipelines.append((parameters, transform))

        transformed_images = []

        for params, func in pipelines:
            img_shape = img.shape

            if self.use_both_lighting:
                # TODO: Untested for batching
                # Expand dims to (B, H, W, 1) for both lighting conditions
                img0 = np.expand_dims(img[:, :, :, 0], axis=[-1])
                img1 = np.expand_dims(img[:, :, :, 1], axis=[-1])

                # Transforms the image according to the current transform
                img0 = func[0](img0, **params[0], is_mask=False)
                img1 = func[1](img1, **params[1], is_mask=False)
                # Outputs (B, H, W, 1)

                # Concatenates the image by the last dimension
                img = np.concatenate([img0, img1], axis=-1)
                # Outputs (B, H, W, 2)
                transformed_images.append((img0, img1))
            else:
                # if both lightings are used but only a single pipeline
                # is used.
                n_images = img.shape[-1]
                if n_images == 1:
                    img = func[0](img, **params, is_mask=False)
                else:
                    img0 = np.expand_dims(img[:, :, :, 0], axis=[-1])
                    img1 = np.expand_dims(img[:, :, :, 1], axis=[-1])

                    img0 = func[0](img0, **params, is_mask=False)
                    img1 = func[0](img1, **params, is_mask=False)

                    # Concatenates the image by the last dimension
                    img = np.concatenate([img0, img1], axis=-1)

                transformed_images.append((img0, img1))

            # The image shape should remain the same
            if img_shape != img.shape and self.enforce_shape:
                raise ValueError(
                    f"The augmentation {func[0].__name__} changed the image shape: "
                    f"original shape: {img_shape}, new shape: {img.shape}"
                )

        return transformed_images

    def __repr__(self):
        repr_str = ""
        for i in range(self.n_pipelines):
            repr_str += f"pipeline {i}: " + " ".join(
                [
                    f"{tf.__name__}: {p}"
                    for tf, p in zip(self.transformations[i], self.parameters[i])
                ]
            )
            repr_str += "\n"

        # Return a readable representation when the image pipeline is printed.
        return repr_str

    @staticmethod
    def build_pipeline(
        pipeline_cfg,
        params: list[int],
        resize_factor: float,
        augment_mask: bool,
        order: list[int] = None,
        use_both_lighting: bool = False,
    ):
        # For custom pipeline ordering
        transform_cfg = []

        pipeline_args = []
        transform_cfg = []

        resize_dict = dict()
        resize_dict["factor"] = resize_factor

        # Iterate over each parameter for both pipelines
        for pipeline_idx in range(params.shape[0]):
            fn_idx = 0
            args = []
            for fn in pipeline_cfg:
                arg_dict = dict()
                for arg in fn["args"]:
                    arg_dict[arg["name"]] = params[pipeline_idx, fn_idx]
                    fn_idx += 1
                args.append(arg_dict)
            if order is not None:
                args = [args[idx] for idx in order[pipeline_idx]]
            args.insert(0, resize_dict)
            pipeline_args.append(args)

            pipeline_functions = [fn["name"] for fn in pipeline_cfg]
            if order is not None:
                pipeline_functions = [pipeline_functions[idx] for idx in order[pipeline_idx]]
            pipeline_functions.insert(0, "resize_by_factor")
            transform_cfg.append(pipeline_functions)

        transformations = [
            [getattr(trans, fn) for fn in pipeline] for pipeline in transform_cfg
        ]

        pipeline = ImagePipeline(
            transformations,
            pipeline_args,
            augment_mask=augment_mask,
            use_both_lighting=use_both_lighting,
        )
        return pipeline
