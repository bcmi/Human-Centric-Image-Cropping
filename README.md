# Human-Centric-Image-Cropping

This is the official repository for the following paper:

> **Human-centric Image Cropping with Partition-aware and Content-preserving Features**
>
> Bo Zhang, Li Niu, Xing Zhao, Liqing Zhang<br>
> Accepted by *ECCV2022*.

We consider a specific and practical application: human-centric image cropping, which focuses on the depiction of a person. 
To this end, we propose a human-centric image cropping method with human-centric partition and important content preservation.
As illustrated in the figure below, the proposed method uses two novel feature designs for the candidate crop: partition-aware feature and content-preserving feature.
The partition-aware feature allows to treat different regions in a candidate crop differently conditioned on the human information, 
and the content-preserving feature helps to preserve the important content to be included in a good crop.

<div align="center">
  <img src='https://github.com/bcmi/Human-Centric-Image-Cropping/blob/main/figures/pipeline.png' align="center" width=800>
</div>


## Usage

1. Download the source code and related image cropping datasets including CPC, GAICD, FCDB, and FLMS datasets. 
The homepages of these datasets have been summarized in our another repository 
[``Awesome-Aesthetic-Evaluation-and-Cropping``](https://github.com/bcmi/Awesome-Aesthetic-Evaluation-and-Cropping).

2. We provide human bounding boxes of the selected human-centric images in above datasets under the folder 
[``human_bboxes``](https://github.com/bcmi/Human-Centric-Image-Cropping/tree/main/human_bboxes).

3. Change the dataset pathes in ``config_GAICD.py`` and ``config_CPC.py``.

4. Build the RoI&RoDAlign libraries. The source code of RoI&RoDAlign is from 
[[here]](https://github.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch).

5. Run ``generate_pseudo_heatmap.py`` to generate pseudo heatmaps for GAICD or CPC dataset. 

6. Run ``train_on_GAICD.py`` or ``train_on_CPC.py`` to train a new model on GAICD or CPC dataset.

### Requirements
Please see [``requirement.txt``](./requirements.txt).

## Other Resources

+ [Awesome-Aesthetic-Evaluation-and-Cropping](https://github.com/bcmi/Awesome-Aesthetic-Evaluation-and-Cropping)

## Acknowledgement<a name="codesource"></a> 

This implementation borrows from [GAICD](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch).

