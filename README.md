<div align="center">
  
# Domain Adaptive Edge-guided DETR for Object Detection

[Full Thesis work here](thesis.pdf)
</div>

This research introduces a novel plug-and-play module tailored for enhancing the generalization capacity of the DETR framework. Leveraging this module, we achieved a substantial 1.9% performance improvement in a competitive state-of-the-art model, elevating detection accuracy from 46.8% to 48.7%. Our approach strategically exploits domain-agnostic information to refine decoder queries, thereby contributing to improved detection performance. This study sheds light on the potential of integrating specialized modules to enhance the capabilities of object detection models like DETR, with implications for advancing performance across diverse domains.

<div align="center" background-color="white">

<img src="https://github.com/marcodavidg/DAE-DETR/assets/11068920/78207bb2-2118-451c-a8f6-ed940b989ebd" alt="drawing" width="700"/>
<br/>
<img src="https://github.com/marcodavidg/DAE-DETR/assets/11068920/7f39b94f-bb6b-4819-831d-f7d83c0be771" alt="drawing" width="700"/>

</div>

## Main Results
This table presents the performance comparison of object detection methods in percentage on the CS (Clear Sky) to Foggy CS domain. Baseline models are indicated with an asterisk(*) sign, and the best results are highlighted in **bold**.


| Method             | Detector | Person | Car  | Truck | Bus  | Train | Mcycle | Bicycle | mAP  |
|---------------------|----------|--------|------|-------|------|-------|--------|---------|------|
| **Faster RCNN***    | Faster RCNN | 26.9 | 35.6 | 18.3 | 32.4 | 9.6   | 25.8   | 28.6    | 26.9 |
| DivMatch     | Faster RCNN | 31.8 | 51.0 | 20.9 | 41.8 | 34.3  | 26.6   | 32.4    | 34.9 |
| SWDA          | Faster RCNN | 31.8 | 48.9 | 21.0 | 43.8 | 28.0  | 28.9   | 35.8    | 35.3 |
| SCDA          | Faster RCNN | 33.8 | 52.1 | 26.8 | 42.5 | 26.5  | 29.2   | 34.5    | 35.9 |
| MTOR          | Faster RCNN | 30.6 | 44.0 | 21.9 | 38.6 | 40.6  | 28.3   | 35.6    | 35.1 |
| CR-DA         | Faster RCNN | 30.0 | 46.1 | 22.5 | 43.2 | 27.9  | 27.8   | 34.7    | 34.2 |
| CR-SW         | Faster RCNN | 34.1 | 53.5 | 24.4 | 44.8 | 38.1  | 26.8   | 34.9    | 37.6 |
| GPA           | Faster RCNN | 32.9 | 54.1 | 24.7 | 45.7 | 41.1  | 32.4   | 38.7    | 39.5 |
| MIC (SADA)    | Faster RCNN | 50.9 | **67.0** | 33.9 | 52.4 | 33.7  | 40.6   | 47.5    | 47.6 |
| **Deformable DETR***| D-DETR   | 38.0 | 45.3 | 16.3 | 26.7 | 4.2   | 22.9   | 36.7    | 28.6 |
| SFA           | D-DETR   | 46.5 | 62.6 | 25.1 | 46.2 | 29.4  | 28.3   | 44.0    | 41.3 |
| MTTrans      | D-DETR   | 47.7 | 65.2 | 25.8 | 49.5 | 33.8  | 32.6   | 46.5    | 43.4 |
| O2net        | D-DETR   | 48.7 | 63.6 | 31.1 | 47.6 | **47.8** | 38.0 | 45.9  | 46.8 |
| DAEG (O2Net) | D-DETR   | **51.9** | 48.6 | **47.7** | **53.3** | 28.3 | **67.9** | **54.9** | **48.7** |
