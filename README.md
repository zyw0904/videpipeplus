## Introduction
`videopipeplus` is a intelligent video analytics framework for edge heterogeneous devices, improved based on the open source project <a href='https://github.com/sherlockchou86/VideoPipe'>VideoPipe</a>，written in C++. It has three key advantages: high hardware adaptability, low development barriers, and external hardware integration. The framework utilizes a modular design to support various edge hardware acceleration devices, allowing flexible adaptation to resource-constrained edge environments. Its model conversion module facilitates cross-device model portability, simplifying deployment. Additionally, the graphical development environment enables users to quickly build video analysis pipelines through intuitive operations, significantly lowering the technical barrier. The framework also supports real-time interaction with external devices, enhancing automation and responsiveness. `Videopipeplus` can be used to build different types of video analysis applications, suitable for scenarios such as video structuring, image search, face recognition, and behavior analysis in many fields.

<p style="" align="center">
  <img src="./docs/1.png" width="480" height="376">
</p>

## Advantages and Features

`Videopipeplus` is similar to NVIDIA's DeepStream and Huawei's mxVision frameworks, however, it has three advantages over other frameworks: high hardware adaptability, low development barriers, and external hardware integration.

### high hardware adaptability

Here is a comparison table:

| **Name**      | **Open Source** | **Learning Curve** | **Supported Platforms** | **Performance** |
|---------------|-----------------|---------------------|--------------------------|-----------------|
| DeepStream    | No              | High                | NVIDIA only              | High            |
| mxVision      | No              | High                | Huawei only              | High            |
| VideoPipe     | Yes             | Low                 | GPU/VPU/TPU              | Medium          |

Here is a hardware hardware architecture of Videopipeplus:

<p style="" align="center">
  <img src="./docs/图片2.png" width="640" height="300">
</p>

`Videopipeplus` can achieve real-time video analysis tasks on a variety of edge heterogeneous devices. A variety of inference engines are integrated within the framework. The inference engines can be selected according to the changes of the scene and model to achieve video analysis on different devices

### low development barriers

### external hardware integration
