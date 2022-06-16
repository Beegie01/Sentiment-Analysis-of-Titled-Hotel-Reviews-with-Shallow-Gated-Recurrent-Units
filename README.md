# Sentiment-Analysis-of-Titled-Hotel-Reviews-with-Shallow-Gated-Recurrent-Units

### PROJECT CONCEPT
This is a research project with the aim of providing a relatively simple, computationally inexpensive approach to sentiment analysis of hotel reviews. To achieve this goal, a well-optimised single hidden-layer gated recurrent units (GRU) was proposed and its performance was compared to that of a baseline model in form of the simple vector machine (SVM).<br><br>

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
Most of the preprocessing was performed using the Scikit-Learn library (Pedregosa et al, 2011), while the GRU classifier was implemented through the deep learning library called Keras (Chollet, 2021). The Adaptive moment (ADAM) optimizer, which runs on a stochastic gradient descent method, is quite resilient to learning rate and other training parameters making it less reliant on hyperparameter optimization (Shewalkar, 2019). Hence, this was chosen as the optimizer for the shallow GRU model. Only the learning rate of ADAM was tuned in this study. The loss function, which computes the distance between predicted and actual values during training, was binary cross-entropy.<br>

#### SVM
For the SVM classifier (with RBF kernel), the input variable consisted of uni-gram, 2-gram, and 3-gram vectors derived from the operation of a count vectorizer. Both the SVM model and the count vectorizer algorithm were implemented through the Scikit-Learn machine learning library.

<br><br>Full details of this research project are documented in this [paper](https://github.com/Beegie01/Sentiment-Analysis-of-Titled-Hotel-Reviews-with-Shallow-Gated-Recurrent-Units/blob/main/Project_report.pdf).
<br>[Here](https://github.com/Beegie01/Sentiment-Analysis-of-Titled-Hotel-Reviews-with-Shallow-Gated-Recurrent-Units/blob/main/Applied_AI_proj.ipynb) is the jupyter notebook for the sentiment analysis of titled hotel reviews with shallow GRU
