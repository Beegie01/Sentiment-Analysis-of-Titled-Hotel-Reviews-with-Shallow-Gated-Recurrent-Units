# Sentiment Analysis of Titled Hotel Reviews with Shallow Gated Recurrent Units

### PROJECT CONCEPT
This is a research project with the aim of providing a relatively simple, computationally inexpensive approach to sentiment analysis of hotel reviews. To achieve this goal, a single hidden-layer gated recurrent units (GRU) was optimised and its performance was compared to that of a baseline simple vector machine (SVM) model.<br><br>
NOTE: This experiment was conducted on the GPU infrastructure of Google Colab virtual platform (https://colab.research.google.com/).<br>
<br><br>Full version of this research project can be found [here](https://github.com/Beegie01/Sentiment-Analysis-of-Titled-Hotel-Reviews-with-Shallow-Gated-Recurrent-Units/blob/main/Project_report.pdf).<br>
<br>[Here](https://github.com/Beegie01/Sentiment-Analysis-of-Titled-Hotel-Reviews-with-Shallow-Gated-Recurrent-Units/blob/main/Applied_AI_proj.ipynb) is the jupyter notebook for the sentiment analysis of titled hotel reviews with shallow GRU<br><br>

### ABSTRACT
The growing volume of existing online reviews today far exceeds the reading capacity of humans. Therefore, there is an urgent need for the introduction of more innovative methods to automate the task of understanding customers’ mindsets through the sentiment analysis of customer reviews (Shi et al 2011). A big part of business success in the hospitality industry (like most industries) is derived from having a good market reputation. Sentiment analysis now provides a way for hospitality service providers (HSPs) to gain a better understanding of how they are perceived in the marketplace by existing and potential customers. In this paper, the focus was on optimizing the hyperparameters of a shallow GRU that can perform sentiment analysis of titled hotel reviews with comparable results to other state-of-the-art algorithms and architectures. As presented in the experimental results, with optimized hyperparameters, a GRU model with one hidden layer improved tremendously in performance and outperformed an SVM classifier. In addition, when tuning the GRU model more attention should be given to the number of units, batch size, and learning rate than the number of layers.<br><br>

### DATASET
![customer reviews of hotels in London](https://user-images.githubusercontent.com/76821049/173886937-80017dff-d71f-4d3d-93e0-1614aedc7ced.png)<br>
Hotel review data used in this research was collected from Opin Rank datasets (Ganesan and Zhai, 2012) comprising titled comments from guests about hotels in many cities worldwide. However, only customer reviews of hotels in London were used in this study. <br>

Due to the absence of sentiment labels, a human annotator was tasked with assigning sentiment labels to the reviews/documents by the emotional meaning of the keywords present in each document. Keywords were determined by selecting only matching words in the topic and body of text of each document. Careful consideration was given to the occurrence of word negation and sarcasm. Examples are shown [here](https://user-images.githubusercontent.com/76821049/173890175-c10a0951-deea-45a7-a7a2-f7c53e4b75d4.png). 
<br>The labelled dataset<br>![Here](https://user-images.githubusercontent.com/76821049/173887463-56e1c396-7f28-4a85-82a7-8ab835d34619.png)<br>8,532 instances were carefully selected from the labeled data with balanced representation from both sentiment classes as shown in the [chart](https://user-images.githubusercontent.com/76821049/173890546-a18d50f5-d674-4982-b659-3560780d8f13.png).
<br> The jupyter notebook showing the sentiment annotation steps can be found [here](https://github.com/Beegie01/Sentiment-Analysis-of-Titled-Hotel-Reviews-with-Shallow-Gated-Recurrent-Units/blob/main/Opin_Rank_annotation_hotel_data.ipynb)<br><br>

### INTUITION-BASED MODEL
![research overview flowchart](https://user-images.githubusercontent.com/76821049/174021226-e30187fc-fd64-4a30-9d7c-46988938b7e5.png)<br>
Because each sentiment label was explicitly assigned by a human agent, this research method can be categorized under the intuition-based approach. After labeling the dataset, the word-to-index and index-to-word lookup dictionaries were created, both containing (in reverse order) pairs of the unique words in the corpus and the index position in the dictionary. Then the data was split into training, validation, and test samples at a ratio of 70: 18: 12 respectively. <br>
Hyperparameters were fine-tuned based on model performance on validation set, while the test set was used to compute performance metrics for the final report. Integer vectors were then created from the input and target variables using the word-to-index dictionary, and one-hot encoder respectively. Then each integer vector was padded to ensure consistency in the input size being fed to the predictive model.  After building and training the classification algorithms, only the hyperparameters of the GRU classifier were fine-tuned.<br><br>

### SUPERVISED MACHINE LEARNING
#### GRU
![image](https://user-images.githubusercontent.com/76821049/174035737-12843595-0688-4ea5-a5f2-aad3c34d6233.png)<br>
Most of the preprocessing was performed using the Scikit-Learn library (Pedregosa et al, 2011), while the GRU classifier was implemented through the deep learning library called Keras (Chollet, 2021). The Adaptive moment (ADAM) optimizer, which runs on a stochastic gradient descent method, is quite resilient to learning rate and other training parameters making it less reliant on hyperparameter optimization (Shewalkar, 2019). Hence, this was chosen as the optimizer for the shallow GRU model. Only the learning rate of ADAM was tuned in this study. The loss function, which computes the distance between predicted and actual values during training, was binary cross-entropy.<br>

#### SVM
For the SVM classifier (with RBF kernel), the input variable consisted of uni-gram, 2-gram, and 3-gram vectors derived from the operation of a count vectorizer. Both the SVM model and the count vectorizer algorithm were implemented through the Scikit-Learn machine learning library.

###	Result Interpretation
#### Training and Validation of Unoptimized GRU
![image](https://user-images.githubusercontent.com/76821049/174037445-fdd9d0b8-2f31-4839-b87e-456f20dd4cb2.png)<br>
The above plot shows a steady rise in validation loss and the reverse for the training loss clearly indicating that overfitting began after epoch 3.<br><br>

#### Training  and Validation of Optimized GRU
![image](https://user-images.githubusercontent.com/76821049/174038667-963a18a8-b9d2-495a-8250-7be68681e26c.png)<br>
The above plot shows a steady drop in training loss, and a tupsy-turvy drop in validation loss. Validation loss was at its minimum at epoch 16 at which, the validation accuracy was slightly higher than the training accuracy implying there was no overfitting.<br><br>

![Performance Metrics](https://user-images.githubusercontent.com/76821049/174040707-1dc982dd-f439-4ffa-b1b2-ef2bcd5e455a.png)
For each metric, the optimized GRU outperformed the SVM, except for recall (negative reviews) and precision (positive reviews) where SVM performed slightly better. It is also important to note the difference between the performance of the GRU model with and without optimization. Therefore, this underlines the importance of fine-tuning hyperparameters in deep learning-based sentiment analysis.<br>
The result presented in this paper proves that a well-tuned shallow GRU has the capacity to compete with or outperform some state-of-the-art algorithms like the SVM. However, as shown by the relatively weaker accuracy of the unoptimized GRU, getting the best performance from the GRU model is relies heavily on selecting the right values for the hyperparameters which often requires research and expertise. The number of units, learning rate, and batch size were highly influential in model performance.<br>

### CONCLUSION
This paper proposes a relatively simple, computationally inexpensive but competitive approach to sentiment analysis of hotel reviews. A lot of sophisticated state-of-the-art approaches have been proposed which are resource-intensive and computationally expensive to implement at a broader scale. In this paper, a shallow GRU with one hidden layer after manually fine-tuning its hyperparameters improved tremendously and outperformed the SVM classifier on five out of 7 performance metrics. This also highlights that fine-tuning hyperparameters is greatly beneficial to the GRU model in the sentiment analysis domain.<br>
In the future, we would like to further improve the performance of the shallow GRU by using a grid search algorithm to automate the optimization process or adding a second hidden layer for more learning capacity. It would also be interesting to see how the combination of the GRU-SVM (RBF kernel) algorithm would perform in sentiment analysis of unlabeled titled hotel reviews due to the promising potential each possesses separately.
