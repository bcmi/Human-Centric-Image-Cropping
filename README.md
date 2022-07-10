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

## Results

In the below figure, we show the source image and the returned best crops by different methods, which demonstrates that our method can perform more reliable content preservation and removal. For example, in the first row, our method preserves more content on the left of human, probably because the person walks right to left, and reduces the top area that may hurt the image composition quality. In the second row, given the opposite face orientations to the first row, our model performs obviously different content preservation on the left/right sides of the human, yielding visually appealing crop. More qualitative and quantitative results are shown in our paper and supplementary.

<div align="center">
  <img src='https://github.com/bcmi/Human-Centric-Image-Cropping/blob/main/figures/qualitative_comparison.png' align="center" width=800>
</div>


## Usage

Here we not only release the code of our method, but also provide the selected human-centric samples in these frequently used image cropping datasets, 
as well as their human bounding boxes under the folder 
[``human_bboxes``](https://github.com/bcmi/Human-Centric-Image-Cropping/tree/main/human_bboxes).

1. Download the source code and related image cropping datasets including CPC, GAICD, FCDB, and FLMS datasets. 
The homepages of these datasets have been summarized in our another repository 
[``Awesome-Aesthetic-Evaluation-and-Cropping``](https://github.com/bcmi/Awesome-Aesthetic-Evaluation-and-Cropping).

2. Change the pathes to above datasets and annotation files in ``config_GAICD.py`` and ``config_CPC.py``.

3. Run ``generate_pseudo_heatmap.py`` to generate pseudo heatmaps for GAICD or CPC dataset. 

4. Install the RoI&RoDAlign libraries following the instruction of [GAICD](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch).

5. Run ``train_on_GAICD.py`` (*resp.*, ``train_on_CPC.py``) to train a new model on GAICD dataset (*resp.*, CPC dataset).

### Requirements
Please see [``requirement.txt``](./requirements.txt).

## Other Resources

+ [Awesome-Aesthetic-Evaluation-and-Cropping](https://github.com/bcmi/Awesome-Aesthetic-Evaluation-and-Cropping)

## Acknowledgement<a name="codesource"></a> 

This implementation borrows from [GAICD](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch).

