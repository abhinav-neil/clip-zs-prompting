# Zero-shot learning & prompting with CLIP

 ## Description
 This repo performs a range of image classification tasks using [CLIP](https://github.com/openai/CLIP).

 CLIP (Contrastive Language-Image Pre-Training) by OpenAI is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

 The datasets used are [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The CIFAR-100 dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. 

 We use CLIP for the following sets of taks:

 ## Zero-shot image classification
 We perform zero-shot classification on CIFAR-10 & CIFAR-100 using CLIP with CLIPB/32 (which is CLIP using a ViT-B/32 backbone). We achieve a top-1 accuracy of 89% on CIFAR-10 and 63.5% on CIFAR-100. 
    
    code_dir=./
    dataset=cifar10 # cifar10, cifar100
    split=test # test, train
    python $code_dir/clipzs.py --dataset $dataset --split $split

 ## Prompting

 ### Text-prompting
 We supply new text prompts to CLIP to make predictions for unseen classes in the data, and visualize the results.

```python
prompt_template="the color of this image is {}"
classes='red blue green'
python $code_dir/clipzs.py --dataset cifar10 --split test --prompt_template "$prompt_template" \
    --class_names $classes --visualize_predictions
```

 ### Visual-prompting
 We create different types of visual prompts for the CLIP models: 
 - a padding prompt of size 30 pixels
 -  a fixed patch prompt of size 1 pixel
 -  a random patch prompt of size 1 pixel
 We then retrain CLIP using each of these prompts and make predictions using the prompted model. The greatest improvement (from the baseline) is observed for the padding prompt, with a top-1 accuracy of 92.3% on CIFAR-10 (test) and 69.6% on the CIFAR-100 (test), after 10 epochs. The other prompts do not yield significant improvements from the baseline.

```python
# training
method=padding # padding, fixed_patch, random_patch
prompt_size=30 # 30 (padding), 1 (fixed patch), 1 (random patch)
epochs=10
patience=2 # early stopping
python $code_dir/main.py --dataset cifar100 --epochs $epochs --method $method --prompt_size $prompt_size \
    --patience $patience

# evaluation (using saved model)
python $code_dir/main.py --dataset cifar100 --evaluate --resume <checkpoint-filename> \
    --visualize_prompt
```

 ## Robustness to noise
 We compare robustness to noise for the various models trained using prompting. We add Gaussian noise to the test sets of CIFAR 10/100 and perform classification using the trained models from the previous step. For both datasets and all 3 types of prompts, the drop in accuracy is between 1–4%, indicating that our models are quite robust to random noise in the data.

 ```
python $code_dir/robustness.py --dataset cifar100 --evaluate --resume <checkpoint-filename> --test_noise
```

 ## Cross-dataset evaluation
 We evaluate the models trained on each dataset (for each visual prompt) on the combination of the 2 datasets, to measure the generalization ability of CLIP. The results are slightly better for the models pretrained on the larger CIFAR-100 dataset, and for the padding prompt, which is what we would expect.

```
python $code_dir/cross_dataset.py --dataset cifar100 --evaluate --resume <checkpoint-filename>
```