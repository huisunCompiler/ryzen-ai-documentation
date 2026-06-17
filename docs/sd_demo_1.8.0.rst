#######################
Stable Diffusion Demo
#######################

Ryzen AI 1.8.0 provides preview demos of Stable Diffusion image-generation pipelines.
The demos cover **Image-to-Image** (ControlNet: Canny, pose, tile, depth; plus Segmind-Vega
i2i without ControlNet) and **Text-to-Image** for SD1.5, SD-Turbo, SDXL-base, SDXL-Turbo,
Segmind-Vega, DreamShaper XL Lightning, SSD-1B, Playground v2.5, FLUX.1-Schnell,
FLUX.2-klein-4B, SD3.0, and SD3.5, with **inpainting** where noted for SD3.0. Weights are
on Hugging Face (AMD ``*-amdnpu`` repos and, where noted, ``stabilityai`` equivalents).

This topic is the **Ryzen AI 1.8.0** GenAI-SD write-up: unified ``run.py`` entry point,
DynRes reference tables, and GenAI-SD runtime defaults. Section order follows the user
guide layout—**Installation Steps**, **Running the Demos** (Image-to-Image with
ControlNet, then Text-to-Image, then Inpainting), and **Running with AMD Stable Diffusion
Sandbox**.

.. _supported-models:

******************
Supported models
******************

The following summarizes default application mode, typical resolution /
dynamic-resolution (DynRes) notes, and the recommended Hugging Face ``model_id`` on
AMD NPU-tuned repos.

.. list-table::
   :header-rows: 1
   :widths: 8 18 12 22 35

   * - Notes
     - Model
     - App
     - Default resolution / DynRes
     - ``model_id`` (AMD Hub; ``stabilityai`` equivalent where applicable)
   * -
     - SD1.5
     - t2i
     - 512x512
     - ``amd/stable-diffusion-1.5-amdnpu``
   * - New
     - SD1.5
     - i2i-canny
     - 512x512
     - ``amd/sd1.5-controlnet-canny-amdnpu``
   * -
     - SD-Turbo (bs1)
     - t2i
     - 512x512
     - ``amd/sd-turbo-amdnpu`` (or ``stabilityai/sd-turbo-amdnpu``)
   * -
     - SDXL-Turbo (bs1)
     - t2i
     - 512x512, 5x DynRes
     - ``amd/sdxl-turbo-amdnpu`` (or ``stabilityai/sdxl-turbo-amdnpu``)
   * -
     - SDXL-base
     - t2i
     - 1024x1024, 20x DynRes
     - ``amd/sdxl-base-amdnpu`` (or ``stabilityai/sdxl-base-amdnpu``)
   * -
     - Segmind-Vega
     - t2i / i2i
     - 1024x1024
     - ``amd/segmind-vega-amdnpu``
   * - New
     - DreamShaper XL Lightning
     - t2i
     - 1024x1024, 20x DynRes
     - ``amd/dreamshaper-xl-lightning-amdnpu``
   * - New
     - SSD-1B
     - t2i
     - 1024x1024, 20x DynRes
     - ``amd/SSD-1B-amdnpu``
   * - New
     - Playground 2.5
     - t2i
     - 1024x1024, 20x DynRes
     - ``amd/playground-v2.5-1024px-aesthetic-amdnpu``
   * - New
     - FLUX.1-Schnell
     - t2i
     - 1024x1024, 20x DynRes
     - ``amd/FLUX.1-schnell-amdnpu``
   * - New
     - FLUX.2-klein-4B
     - t2i
     - 1024x1024
     - ``amd/FLUX.2-klein-4B-amdnpu``
   * -
     - SD3.0
     - t2i / ControlNet / Depth / Canny / Pose / Tile
     - 512x512 (t2i default), 20x DynRes where applicable
     - ``amd/stable-diffusion-3-medium-amdnpu`` (or ``stabilityai/stable-diffusion-3-medium-amdnpu``)
   * - New
     - SD3.0
     - i2i-inpainting
     - 1024x1024
     - ``amd/stable-diffusion-3-medium-amdnpu`` (or ``stabilityai/stable-diffusion-3-medium-amdnpu``)
   * -
     - SD3.5
     - t2i / ControlNet
     - 512x512 (t2i default), 20x DynRes where applicable
     - ``amd/stable-diffusion-3.5-medium-amdnpu`` (or ``stabilityai/stable-diffusion-3.5-medium-amdnpu``)
   * -
     - SD3.5-ControlNet (Canny)
     - i2i-canny
     - 512x512, 20x DynRes
     - Uses SD3.5 AMD weights plus SD3.0 Canny ControlNet assets (see below)

The supported-model column "DynRes" counts how many dynamic-resolution presets a
pipeline exposes (for example 5x or 20x). The following material spells out which
width x height pairs are in scope.

.. _dynres:

.. rubric:: Dynamic resolution (DynRes)

**Original dynamic-resolution requirements**

These are the classic fixed pairs (per model class) before the expanded preset grid.

* **SDXL-Turbo:** 512x512, 288x512, 512x288, 384x512, 512x384.
* **All other models** (that advertise DynRes in the table above): 1024x1024,
  576x1024, 1024x576, 768x1024, 1024x768.

**New dynamic-resolution combinations**

Additional width x height combinations are supported with **output batch size 1**.
Linear dimensions are **rounded up to a multiple of 32** where the stack requires
aligned tensor geometry.

The following grid lists **base 1:1** square resolutions in the header row; each
row is a target **aspect ratio** expressed as width x height for that base.

.. list-table:: New DynRes combinations (batch size 1; align to multiple of 32 as required)
   :header-rows: 1
   :widths: 14 18 18 18 22

   * - Aspect (WxH)
     - Base 512x512
     - Base 640x640
     - Base 768x768
     - Base 1024x1024
   * - 1:1
     - 512x512
     - 640x640
     - 768x768
     - 1024x1024
   * - 4:3
     - 512x384
     - 640x480
     - 768x576
     - 1024x768
   * - 3:4
     - 384x512
     - 480x640
     - 576x768
     - 768x1024
   * - 16:9
     - 512x288
     - 640x384
     - 768x448
     - 1024x576
   * - 9:16
     - 288x512
     - 384x640
     - 448x768
     - 576x1024

Public model pages follow the Hugging Face model license (HF LIC) for each repo.

**SD3.5 Canny ControlNet setup:** copy the SD3.0 Canny ControlNet files into the SD3.5
model layout as required by your GenAI-SD tree (per release notes), then run the Canny
example with ``--model_id amd/stable-diffusion-3.5-medium-amdnpu``.

******************
Installation Steps
******************

1. Ensure the latest version of Ryzen AI and NPU drivers are installed. See
   `Ryzen AI installation <https://ryzenai.docs.amd.com/en/latest/inst.html>`_.

2. The GenAI-SD folder is located in the RyzenAI installation tree. Navigate to the
   folder and run the following command:

   .. code-block:: powershell

      conda activate ryzen-ai-1.8.0
      cd "$env:RYZEN_AI_INSTALLATION_PATH\GenAI-SD"

3. Models are downloaded from Hugging Face on first use and cached under
   ``GenAI-SD\models`` or ``$env:RYZENAI_GENAI_SD_MODELS_ROOT``. See `Supported models`_
   for the full list; primary entry points include:

   - `SD1.5 <https://huggingface.co/amd/stable-diffusion-1.5-amdnpu>`_
   - `SD1.5 ControlNet Canny <https://huggingface.co/amd/sd1.5-controlnet-canny-amdnpu>`_
   - `SD-Turbo <https://huggingface.co/amd/sd-turbo-amdnpu>`_
   - `SDXL-Turbo <https://huggingface.co/amd/sdxl-turbo-amdnpu>`_
   - `SDXL-base <https://huggingface.co/amd/sdxl-base-amdnpu>`_
   - `Segmind-Vega <https://huggingface.co/amd/segmind-vega-amdnpu>`_
   - `DreamShaper XL Lightning <https://huggingface.co/amd/dreamshaper-xl-lightning-amdnpu>`_
   - `SSD-1B <https://huggingface.co/amd/SSD-1B-amdnpu>`_
   - `Playground v2.5 1024px <https://huggingface.co/amd/playground-v2.5-1024px-aesthetic-amdnpu>`_
   - `FLUX.1-Schnell <https://huggingface.co/amd/FLUX.1-schnell-amdnpu>`_
   - `FLUX.2-klein-4B <https://huggingface.co/amd/FLUX.2-klein-4B-amdnpu>`_
   - `SD3.0 <https://huggingface.co/amd/stable-diffusion-3-medium-amdnpu>`_
   - `SD3.5 <https://huggingface.co/amd/stable-diffusion-3.5-medium-amdnpu>`_

******************
Running the Demos
******************

Activate the conda environment (see also `Installation Steps`_):

.. code-block:: powershell

   conda activate ryzen-ai-1.8.0

Optionally, set the NPU to high performance mode to maximize performance:

.. code-block:: powershell

   xrt-smi configure --pmode performance

Refer to `xrt-smi configure <https://ryzenai.docs.amd.com/en/latest/xrt_smi.html#xrt-smi-configure>`_
in the Ryzen AI documentation for additional options.

From the ``GenAI-SD\test`` directory unless noted otherwise. All examples use the unified
entry point ``run.py`` and pass ``--model_id`` with the Hugging Face model identifier.

.. _sd3-env:

.. rubric:: Image-to-Image with ControlNet

The image-to-image demo generates images from a **prompt** plus a **control image**
(ControlNet types such as Canny, pose, tile, or depth for SD3.x, selected with ``-C``).
SD3.x often defaults to 512x512; override resolution with ``-H`` and ``-W`` as in the
examples below. SD3.x DynRes presets are summarized in `Supported models`_ and
`Dynamic resolution (DynRes) <dynres_>`_.

**GenAI-SD runtime:** ``DD_PLUGINS_ROOT`` is assigned **automatically** when the GenAI-SD
stack runs (for example when you invoke ``run.py``). Export or set ``DD_PLUGINS_ROOT``
yourself only for a **custom** tree where plugin or dependency paths differ from what the
runtime discovers in the stock Ryzen AI / GenAI-SD layout.

To run a minimal Canny example:

.. code-block:: powershell

   python run.py -C canny --model_id amd/stable-diffusion-3-medium-amdnpu

The demo can use ``.\ref\canny.jpg`` as the control image unless you override
``--control_image_path``. Outputs go to ``generated_images`` unless you set
``--output_path``.

**SD1.5 ControlNet Canny (i2i-canny)**

.. code-block:: powershell

   python .\run.py --model_id amd/sd1.5-controlnet-canny-amdnpu

**Segmind-Vega (i2i, no ControlNet path)**

.. code-block:: powershell

   python .\run.py --model_id amd/segmind-vega-amdnpu --control_image_path .\assets\controlimg_input_1024x1024.png --strength 0.95

**SD3.0 ControlNet (canny / pose / tile / depth)**

.. code-block:: powershell

   python .\run.py -C canny --model_id amd/stable-diffusion-3-medium-amdnpu --prompt "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image" -H 1024 -W 1024 --control_image_path .\ref\canny.jpg -n 50
   python .\run.py -C pose --model_id amd/stable-diffusion-3-medium-amdnpu --prompt "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image" -H 1024 -W 1024 --control_image_path .\ref\pose.jpg -n 50
   python .\run.py -C tile --model_id amd/stable-diffusion-3-medium-amdnpu --prompt "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image" -H 1024 -W 1024 --control_image_path .\ref\tile.jpg -n 50
   python .\run.py -C depth --model_id amd/stable-diffusion-3-medium-amdnpu -H 1024 -W 1024 --control_image_path .\assets\depth.jpeg -n 50

**SD3.5 ControlNet Canny (i2i-canny)**

After copying SD3.0 Canny ControlNet into the SD3.5 layout as required:

.. code-block:: powershell

   python .\run.py -C canny --model_id amd/stable-diffusion-3.5-medium-amdnpu --prompt "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image" -H 1024 -W 1024 --control_image_path .\ref\canny.jpg -n 50

.. rubric:: Text-to-Image

The text-to-image demo generates images from **text prompts only** (no control image).
It covers SD1.5 (512-class), SD-Turbo and SDXL-Turbo (512-class), SDXL-base, Segmind-Vega,
DreamShaper XL Lightning, SSD-1B, Playground v2.5, FLUX.1-Schnell, FLUX.2-klein-4B, and
SD3.0 / SD3.5 with ``-C None``. Use ``-H``, ``-W``, and ``-n`` when the pipeline supports
them. ``DD_PLUGINS_ROOT`` for SD3.x is handled by the GenAI-SD runtime; see
`Image-to-Image with ControlNet <sd3-env_>`_ above.

From ``GenAI-SD\test``, run the following to exercise each checkpoint (same pattern as
the published guide, with a single ``run.py`` entry point):

.. code-block:: powershell

   python .\run.py --model_id amd/stable-diffusion-1.5-amdnpu
   python .\run.py --model_id amd/sd-turbo-amdnpu
   python .\run.py --model_id amd/sdxl-turbo-amdnpu
   python .\run.py --model_id amd/sdxl-base-amdnpu
   python .\run.py --model_id amd/segmind-vega-amdnpu
   python .\run.py --model_id amd/dreamshaper-xl-lightning-amdnpu
   python .\run.py --model_id amd/SSD-1B-amdnpu
   python .\run.py --model_id amd/playground-v2.5-1024px-aesthetic-amdnpu
   python .\run.py --model_id amd/FLUX.1-schnell-amdnpu
   python .\run.py --model_id amd/FLUX.2-klein-4B-amdnpu
   python .\run.py -C None --model_id amd/stable-diffusion-3-medium-amdnpu -H 1024 -W 1024 -n 50
   python .\run.py -C None --model_id amd/stable-diffusion-3.5-medium-amdnpu -H 1024 -W 1024 -n 50

Custom prompts can be supplied with ``--prompt``. For example:

.. code-block:: powershell

   python .\run.py --model_id amd/stable-diffusion-1.5-amdnpu --prompt "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details"

.. rubric:: Inpainting

**SD3.0 (``-C Inpainting``)** uses a base image and ``--control_mask_path`` (URLs or local
paths). This extends the image-conditioned flows above with an explicit mask channel.

.. code-block:: powershell

   python .\run.py --model_id amd/stable-diffusion-3-medium-amdnpu -C Inpainting --prompt "A cat is sitting next to a puppy" --n_prompt "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW" -n 28 --controlnet_conditioning_scale 0.95 --control_image_path "https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting/resolve/main/images/dog.png" --control_mask_path "https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting/resolve/main/images/dog_mask.png" --seed 42

*****************************************
Running with AMD Stable Diffusion Sandbox
*****************************************

AMD SD Sandbox is a framework for running Stable Diffusion (SD) models accelerated by
AMD Ryzen AI hardware. It provides an easy-to-use interface for evaluating, comparing,
and deploying multiple SD pipelines.
Please go to the `AMD SD Sandbox GitHub repository <https://github.com/amd/sd-sandbox>`_
for more information.

..
   ------------
   #####################################
   License
   #####################################

   Ryzen AI is licensed under `MIT License <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_.
   Refer to the `LICENSE File <https://github.com/amd/ryzen-ai-documentation/blob/main/License>`_
   for the full license text and copyright notice.
