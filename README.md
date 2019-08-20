# Disease detection Project using Chest X-ray Database
This project aims at a new chest X-ray database, namely “ChestX-ray8”, which comprises 108,948 frontal view X-ray images of 32,717 unique patients with the textmined 14 disease image labels (where each image can have multi-labels), from the associated radiological reports using natural language processing.
## Abstract
Chest X-Rays are the most reliable radiobiological imprints of patients, widely used to efficiently diagnose an array of common thoracic diseases. For too long, vast accumulations of image data and their associated diagnoses have been stored in the Picture Archiving and Communication Systems (PACS) of several hospitals and medical institutions. In the meanwhile, data-hungry Deep Learning systems lie in wait of voluminous databases just like these, at the cusp of fulfilling the promise of fully-automated and accurate disease diagnosis. Through this project, we hope to unite one such vast database, the “ChestX-ray8" dataset, with powerful Deep Learning Systems, to automate the diagnosis of 14 common kinds of lung diseases. Currently, we will be focusing on three kinds of diseases to start with. 

## Introduction 
### Why Deep Learning for Disease Diagnostics?
  * Convolutional Neural Networks, which form the soul of Deep Learning systems, are designed with the assumption that they will be processing images, according to computer science experts at Stanford University, allowing the networks to operate more efficiently and handle larger images.
  * As a result, some CNNs are approaching – or even surpassing – the accuracy of human diagnosticians when identifying important features in diagnostic imaging studies.
  * In June of 2018, a study in the Annals of Oncology showed that a convolutional neural network trained to analyze dermatology images identified melanoma with ten percent more specificity than human clinicians.
  * Researchers at the Mount Sinai Icahn School of Medicine have developed a deep neural network capable of diagnosing crucial neurological conditions, such as stroke and brain hemorrhage, 150 times faster than human radiologists. The tool took just 1.2 seconds to process the image, analyze its contents, and alert providers of a problematic clinical finding.

All these reasons show that Speed and Accuracy are characteristic features of ‘Deep Learning- driven’ Medical Diagnostics. This indicates the potential for getting optimum results in exploring this field.

### Why does Medical Diagnosis need Deep Learning?
  * **More affordable treatment:** Diagnosis is faster and more accurate with automation; doctors will be able to recommend the right medicine to patients before illnesses aggravate to require more expensive treatment options.
  * **Safer solutions:** More accurate diagnosis means there’s a lower risk of complications associated with patients receiving ineffective or incorrect treatment.
  * **More patients treated:** By reducing the time it takes to complete a diagnosis, laboratories can perform more tests, covering a much larger number of patients in much lesser time. 
  * **Addressing the global ‘Physician Shortage’:** This is a growing concern in many countries around the world, due to a growing demand for physicians that outmatches the supply. The World Health Organization (WHO) estimates that there is a global shortage of 4.3 million physicians, nurses, and other health professionals. The shortage is often starkest in developing nations due to the limited numbers and capacity of medical schools in these countries. Additionally, rural and remote areas also commonly struggle with a physician shortage the world over.

So, the growing need for more qualified medical personnel worldwide is like an incomplete jigsaw puzzle.  Deep Learning systems have the potential to finish this puzzle once and for all.

This is exactly why we feel that the quest to build such a system is truly relevant to the needs of society at present. There is no denying the fact that with an ever-growing global population, we are going to need alternative AI-based medical personnel to assist us in achieving the sustainable development goal of providing “**A**ffordable,**A**ccurate and **A**dequate Healthcare for All”. 

## Methodology
Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature-maps of all preceding layers are used as inputs, and its own feature-maps are used as inputs into all subsequent layers. 

Given our time constraint, we chose to divide ourselves into sub-teams corresponding to the demarcated phases of this project. The benefit of this approach was a smooth iterative process - where we could have quick adjustments from previous phases to facilitate better outcomes for a subsequent phase - and speed due to parallel implementations at the various phases.
We also adopted a scrum model, where the entire period was a sprint, and stand-up sessions within at most 2-day intervals served for a progress update and team re-orientation where it was necessitated.
Finally, our team model allowed us to use initial cycles as an exploration to inform our final data, preprocessing and model strategies. With each cycle, we redefined our goals while maintaining the general objective of facilitating faster lung X-ray assessments. 

![Stages for Project](https://user-images.githubusercontent.com/37798451/63227239-07e9d980-c202-11e9-8bfd-29c635a12956.png)

### Sampling

The dataset was highly imbalanced, a high value in the distribution of 60361 and low of 110, and huge in size for our timeline and we had to resort it to using a well-represented sample. We eventually scaled down on the dataset to [12000] from [112000]. 

The initial distribution for images with single class labels is given below.

Labels | Distributions
------------ | -------------
No Finding | 60361
Infiltration | 9547
Atelectasis | 4215
Effusion | 3955
Nodule | 2705
Pneumothorax | 2194
Mass | 2139
Consolidation | 1310
Pleural_Thickening | 1126
Cardiomegaly | 1093
Emphysema | 892
Fibrosis | 727
Edema | 628
Pneumonia | 322
Hernia | 110

We initially tested our strategies for classifying all 15 conditions and redefined our goals through exploration, until in addition to identifying Healthy X-rays (No Finding), the model could more accurately classify two conditions, Cardiomegaly and Effusion.

For our final round, and based on clinical considerations, we proceeded with two sets:
* Version 4.1 - for images taken using both antero-posterior position (AP) and postero-anterior position (PA).
* Version 4.2 - containing images taken using PA for Effusion and healthy (No finding) classes, and AP and PA for cardiomegaly. 

The exception was made for the latter due to inadequate PA images.

The distribution for Version 4.1 finally settled at:
* (AP + PA) Cardiomegaly – 1093
* (AP + PA) No Finding – 1500
* (AP + PA) Effusion - 1500

The distribution for Version 4.2 finally settled at:
* (AP + PA) Cardiomegaly – 1093
* (PA) No Finding – 1500
* (PA) Effusion - 1500

### Preprocessing
Given that our dataset covered medical conditions, the manifestations of which could be present at edges of X-ray images, we exercised caution with our transformations. After several deliberations and clinical considerations, we resorted to avoiding cropping and extreme random rotations and kept to these:
* Random Rotation: within the angle range -10 to 10, with expansion enabled
* Resize: to a size of 224 by 224 to match our densenet model input size
* Random Horizontal Flip
* Random Vertical Flip
* Conversion To Pytorch floatTensor type
This minimalistic approach was our shot at preserving as much information that would be clinically relevant for diagnosing our target conditions.

### Modelling
We had a run with densenet161, and resNext50 during our model staging to assess and compare performances, before finally settling with densenet161.
The modeling stage was characterized by several iterative cycles that called for new sampling and processing strategies on demand. Notwithstanding, the team model facilitated swift responses so that we could re-orient quickly without disrupting overall progress.
We also had the technical expertise that allowed us to try novel activation functions - namely mila, mish and beta mish – which we believe contributed greatly to our results, in addition to hyperparameter tunings.

For activation Functions, we are using β-Mish and Mila. 
**β-Mish** is an uni-parametric activation inspired by the Mish activation function - when β=1, β-Mish becomes the standard version of Mish - and can be mathematically represented using the function:

![B-mish](https://user-images.githubusercontent.com/37798451/63227800-20f58900-c208-11e9-8a8b-3ee5f425e086.PNG)

If β=1.5, the function ranges from  ≈-0.451103 to ∞. For most benchmarks, β was set to be 1.5.
![Mish3](https://user-images.githubusercontent.com/37798451/63227815-569a7200-c208-11e9-9412-b802fe7bf20f.png)

**Mila** is an uniparametric activation function inspired from the Mish Activation Function. The parameter β is used to control the concavity of the Global Minima of the Activation Function where β=0 is the baseline Mish Activation Function. Varying β in the negative scale reduces the concavity and vice versa. β is introduced to tackle gradient death scenarios due to the sharp global minima of Mish Activation Function.

The mathematical function of Mila is shown as below:
![Mila](https://user-images.githubusercontent.com/37798451/63227901-4df66b80-c209-11e9-8e8b-1ab785410177.PNG)

### Encryption of model and dataset
Healthcare data is particularly sensitive and if we face the risk of exposing sensitive patient data. For ensuring security and privacy of the dataset, we have implemented encrypted learning. To ensure encryption of model is done on demand and portability of classes we have implemented this in the below manner.
* We encrypt patient data before it reaches our model. 
* We make the model available as a service, and we protect our intellectual rights, as regards the gradients and model parameters.

Below are the configurations to achieve above two cases
* Name Of Class: ModelEncryptor
* Attributes: shares(shareholders), model(encrypted model)
* Methods: encrypt_data(encrypts image data to be classified), predict(classifies the image)

## Results

### *Approach 1:* AURORA: beta-mish, mila and densenet 161 --> 82.3%
This approach is tested with two possibilities of the dataset as mentioned below. 
#### Dataset 1: Mixed Dataset (PA and AP) using a β-Mish activation function 
Test Loss: 0.541957

Label | Accuracy
------------ | -------------
Cardiomegaly | 83.000% (83/100)
Effusion | 79.000% (79/100)
No Finding | 76.000% (76/100)
Overall | 79.3333% (238/300)

### Dataset 2: PA only using Mila activation function
Test Loss: 0.456018

Label | Accuracy
------------ | -------------
Cardiomegaly | 89.000% (89/100)
Effusion | 83.000% (83/100)
No Finding | 75.000% (75/100)
Overall | 82.3333% (247/300)

### *Approach 2:* ATLAS: resNext50, traditional ReLU + Softmax combination --> accuracy 47%
In this approach, we have used resNext50 instead of densenet161 to test if helps us to acheive better accuracy than other models. It also has hidden layers for ReLU and softmax algorithms. 

### *Approach 3:* ARMADILLO:  densenet161, mila activation --> accuracy 61.932%
Here, we have used the only Mila as an activation function to train the model along with densenet161. 
#### Dataset 1: Mixed Dataset (PA and AP) using a β-Mish activation function
Validation accuracy: 61.932
Test Loss: 0.915
Test Accuracy: 61.3

Label | Accuracy
------------ | -------------
Cardiomegaly | 25%
Effusion | 80%
No Finding | 66%

_**So based on the above results, the most accurate model is *AURORA* which use densenet161, beta-mish and Mila as its activation function helping us to achieve an accuracy of 82.3%.**_

## Discussion
We ran into significant difficulties in our attempt to find a working approach towards processing the Chest X-Ray dataset and training a successful deep learning model. While many factors can be taken into account for this complication, the root cause can be traced back to the original dataset itself, which contains many imperfections and deficiencies that posed numerous challenges for the Data Acquisition, Pre-processing and Modelling teams alike. 

All of those shall be explored and clarified in-depth for our readers in this section, and the limitations we came across will be covered in the next section.
* **Chest X-Ray dataset is extremely imbalanced when it comes to the distribution of the number of instances for each class**: _For example,_ there is a huge disparity between the number of the top class – No Finding with more than 60,000 instances and the bottom one – Hernia with mere 110 instances. 
        This created a difficult situation for us where we had to make a decision on how we should further process and transform our data before feeding to the model.
  
* **X-ray images in our dataset were taken from two different positions: PA/posterior anterior and AP/anterior posterior**:  
This especially held true for “Effusion” class, with the number of images captured in each position almost equal. Since many of the diseases in our dataset were sensitive to such positions, we debated on whether we should split each class into two smaller ones: AP and PA or more ideally, only retain PA pictures. <br>
The latter approach, nevertheless, had one major disadvantage: we need to take into account the number of pictures in the PA position, which in many cases there were simply not enough of them. In the end, considering the limited number of PA images we had for “cardiomegaly”, we decided to include both PA and AP pictures in each class, and this would definitely have a negative impact on the accuracy of our model.

* **How we should proceed with data augmentation**: 
Given the nature of X-ray images for lung diseases, even a small rotation or central cropping could end up leaving out important diagnosis information and lead to the difficulty of generalization. Having researched extensively to gain a deep understanding of the images, the size of our dataset, and the reported performance of past models, after lengthy discussions and countless trial and error attempts, we finally came to an agreement: reduce the number of classes in the wrangled and modified dataset to only 3: CardioMegaly, Effusion, and No finding, with the number of sampled instances for each class, became much more balanced and therefore prevented biases from occurring when being trained by the model.

* **X-ray images in the dataset were missing some crucial information typically used for diagnosing diseases**: 
After consulting with the subject matter expert Olivia in our team, we found that in particular, for conditions that tend to look similar in X-ray images such as “Mass”, “Infiltration” and “Pneumonia”, in real life doctors would rely on other factors such as white count and temperature to make a formal diagnosis. Unfortunately, that information was not available in our dataset, which in turn can be problematic for whatever deep learning model being trained and deployed to classify conditions. In fact, in the original paper, all the fore-mentioned diseases performed poorly and recorded a high prediction error rate from the model, which helped to reinforce our earlier finding.

* **Disease labels for the X-ray chest dataset were not done manually, but rather through a number of different NLP techniques when constructing the image database**: 
Although the authors have gone the extra mile to mitigate the risk of wrong labeling by crafting a variety of custom rules and then applying them in the pre-processing step, the problem of defective labels was not entirely eliminated. Specifically, we can refer to “Table 2. Evaluation of image labeling results on OpenI dataset.” on page 4 of the paper for a detailed assessment on this phenomenon. The vast majority of classes exhibit varying degrees of being mislabeled, from Effusion with 0.93 precision rate to Infiltration with a modest score of 0.74. As a consequence, this has considerably limited our model’s ability to train on the dataset and later classify disease labels with great accuracy, given that the data was not without flaw right from the beginning.

## Limitations
Few Limitations that we found while working on this project one of them explained in above discussions:
* Negative impact on the accuracy of our model due to the inclusion of both PA and AP pictures in each class.
* For data augmentation, we need an alpha(Fluorescence) channel to avoid dismissal of key info which was creating difficulty of generalization. So we reduced to 3 classes and modified the dataset to prevent biases from occurring when being trained by the model.
* Considering the picture quantities we have for cardiomegaly, we decided to still include both AP and PA pictures. This may lower the accuracy of our model.

## Conclusion
## Recommendations
The team would like to extend the project to institutions where aid to diagnosis is of utmost importance. 
* Currently the project is limited to the public domain dataset and to the best-effort analyses of health records via natural language processing. The idea here is to improve the current accuracy of the model by augmenting it with real-world datasets which are available from medical institutions. 
* Due to the sensitive nature of these datasets and with intentions of privacy, naturally these are currently being kept private. With the power of federated learning, we can adopt a strategy where medical institutions would not need to relent their private datasets to a central server, which might lead to privacy leakage. Instead, we open an interface for them to feed the data within the institution, train the model on-site and only transmit gradients and other model information to the central server which we have access and do model aggregation accordingly. 
* Firstly, we coordinate with medical institutions to install Internet-enabled devices on-premise. For this, we think of Raspberry Pi 3 devices, small, lightweight and powerful enough to do the tasks we need for local training. We plan on creating a headless setup to each Raspberry Pi devices with web server connected on their local network. This web server can be accessed by representatives in-house to feed X-ray data and other relevant information pertinent to local training. We make sure that the data to be fed to locally matches the global requirements for model improvement. 
* We can send model definition to each Raspberry Pi’s installed remotely. We then orchestrate model updates on-demand using the power of PySyft’s secure model parameter aggregation, leveraging mathematical techniques per actor to encrypt model information so trusted aggregators cannot glean on raw gradients sent by federated nodes.
Of course, there needs to be full coordination with hospitals, clinics and radiologic facilities who have quality datasets to join in our planned IoT-enabled space specifically for this use case. In return, we enable an intuitive interface to help doctors in diagnosis. 
* For improving encrypted deep learning in this project, we would try to improve and tweak class where we can customize precision.

Due to the sensitive nature of these datasets and with intentions of privacy, naturally, these are currently being kept private. 
## Appendix
https://colab.research.google.com/drive/1nub56-UfvlovgWP7oSC5850HdNIbOFQu 

## Collaborators
Members | Slack Handle
------------ | -------------
Victor Mawusi Ayi | @ayivima
Anju Mercian | @Anju Mercian
George Christopoulos | @George Christopoulos
Ashish Bairwa  | @Stark
Pooja Vinod | @Pooja Vinod
Ingus Terbets | @Ingus Terbets
Alexander Villasoto | @Alexander Villasoto
Olivia Milgrom | @Olivia
Tuan Hung Truong | @Hung
Marwa Qabeel | @Marwa
Shudipto Trafder | @Shudipto Trafder
Aarthi Alagammai | @Aarthi Alagammai
Agata | @Agata [OR, USA]
Kapil Chandorikar | @Kapil Chandorikar
Archit | @Archit
Cibaca Khandelwal | @cibaca
Oudarjya Sen Sarma | @Oudarjya Sen Sarma
Rosa Paccotacya | @Rosa Paccotacya

## References

https://arxiv.org/pdf/1705.02315.pdf

http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf

https://www.kaggle.com/ingusterbets/nih-chest-x-rays-analysis

https://github.com/digantamisra98/Mila

https://github.com/digantamisra98/Beta-Mish

https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737

https://nihcc.app.box.com/v/ChestXray-NIHCC/file/219760887468

https://link.springer.com/chapter/10.1007/978-3-540-75402-2_4

https://openreview.net/pdf?id=rkBBChjiG

https://arxiv.org/pdf/1711.05225

