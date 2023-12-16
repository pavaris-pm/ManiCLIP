# ManiCLIP (Colab-Compatible version)
- This is a colab compatible version, you can directly check on `maniclip_pipeline.ipynb` (open in colab function supported, no need to load it)
- dedicated training and architecture setup by Mr. Pavaris Ruangchutiphophan (`github`: pavaris-pm)

-------------------------------------------

## Official PyTorch implementation

![Teaser image](teaser.png)

**ManiCLIP: Multi-Attribute Face Manipulation from Text**   
Hao Wang, Guosheng Lin, Ana García del Molino, Anran Wang, Jiashi Feng, Zhiqi Shen   
[**Paper**](https://arxiv.org/abs/2210.00445)

Abstract: *In this paper we present a novel multi-attribute face manipulation method based on textual descriptions. Previous text-based image editing methods either require test-time optimization for each individual image or are restricted to single attribute editing. Extending these methods to multi-attribute face image editing scenarios will introduce undesired excessive attribute change, e.g., text-relevant attributes are overly manipulated and text-irrelevant attributes are also changed. In order to address these challenges and achieve natural editing over multiple face attributes, we propose a new decoupling training scheme where we use group sampling to get text segments from same attribute categories, instead of whole complex sentences. Further, to preserve other existing face attributes, we encourage the model to edit the latent code of each attribute separately via an entropy constraint. During the inference phase, our model is able to edit new face images without any test-time optimization, even from complex textual prompts. We show extensive experiments and analysis to demonstrate the efficacy of our method, which generates natural manipulated faces with minimal text-irrelevant attribute editing.*

## Dataset

During the training phase, we do not need any data, except the 40-category face [attributes](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (list_attr_celeba.txt). During the testing phase, the text data can be obtained from [Multi-Modal-CelebA-HQ](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset) (text.zip).

## Pretrained models

The pretrained models can be downloaded from this [link](https://hkustgz-my.sharepoint.com/:f:/g/personal/haowang_hkust-gz_edu_cn/EjFaiuxDA9ZJk6nBCyLBLroBivUDY4CjI9tM9TNpAzI2PA).

## Training

You can train new face editing networks using `train.py`.

```.bash
python train.py --epochs 30 --loss_id_weight 0.05 --loss_w_norm_weight 0.1 --loss_clip_weight 1.0 --loss_face_norm_weight 0.05 --loss_minmaxentropy_weight 0.2 --loss_face_bg_weight 1 --task_name name --decouple --part_sample_num 3
```

## Generation

To generate edited images based on language:

```.bash
python generate.py --model_path pretrained/pretrained_edit_model.pth.tar --text "this person has grey hair. he has mustache." --gen_num 5
```

## Reference
If you find this repository useful, please cite:
```
@article{wang2022maniclip,
  title={ManiCLIP: Multi-Attribute Face Manipulation from Text},
  author={Wang, Hao and Lin, Guosheng and del Molino, Ana Garc{\'\i}a and Wang, Anran and Feng, Jiashi and Shen, Zhiqi},
  journal={arXiv preprint arXiv:2210.00445},
  year={2022}
}
```
-------------------------------------------

## FAQs 
- always read it before asking up any question, if it is already in here, i will not answer it even you contact me about it

### Q1.) Did this notebook is an implementation of ManiCLIP ?

`Ans`: Yes, however, i'm mainly coding in GitHub codespace so that whatever changes i've made will be pushed directly to the repository and will use Google colab to execute the script instead. It is waste of my time to implement it on both environment. Therefore, if you have any question, just lookup the code in the repo, i already leave a comment on it already.

---------------------------------------


### Q2.) I couldn't run it on my environment, how could i do it ?

`Ans`: if you're working on codespace e.g. github codespace, vscode etc.. I recommend you to create a new environment via `env` command by create it from `environment.yaml` file that on this repo. Since the version of `torchvision` library is somewhat needed a specific version (many GANs implementation mainly rely on `vutils` function), for colab, please make sure that you've clone the correct repo (branch name: `for-colab`)



```cli
!git clone --branch for-colab https://github.com/pavaris-pm/ManiCLIP.git
```



-----------------------------------------


  ### Q3.) What is an expected result of this implementation ?

  `Ans`: this is one of the new way of face editing task since many models always handle with only single-attribute only (except with conditional gan that seems much on give a condition, however, its training scheme always look heavy if we training it on the attributes/latent as always) with that ManiCLIP coming out to handle multi-face attribute with a very new training scheme (like think outside-the-box manner)


  Apart from that, i projected to improve it in order to handle with low-resource language so that it will be accessed by most of people to build their own generation models from now on since i saw a possible implementation to get this done, and now i', currently working on it (it'll be done if i'm not that lazy 😭).


  -----------------------------------------
  
   ### Q4.) since you pull some data from Google drive, will the resource on the google drive published?
   ```python
  from google.colab import drive
  drive.mount('/content/drive')
   ```

  `Ans`: No. I will not share all of my resources used to train the model. Everything buildup in this repo can be followed based on the `readme.md` file. Just follow it can replicate the paper as well. 

  -----------------------------------------

  ### Q5.) What if i have a question about this model ?

  `Ans`: i'm a very active developer on github, any questions coming up from this implementation. It is better to open up an issue on my repo and do not forget to `@pavaris-pm` for notify me to read it, i will answer it immediately.

  -----------------------------------------
