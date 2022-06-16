# Sentiment-Analysis-of-Titled-Hotel-Reviews-with-Shallow-Gated-Recurrent-Units

### PROJECT CONCEPT
This is a research project with the aim of providing a relatively simple, computationally inexpensive approach to sentiment analysis of hotel reviews. To achieve this goal, a well-optimised single hidden-layer GRU was proposed and its performance was compared to that of a baseline model in form of the simple vector machine (SVM).<br><br>

### BACKGROUND
#### Sentiment Annotation
When customers submit reviews, they generally do not follow a defined pattern or writing style which makes it difficult to automatically detect the emotional dimension of the comment (document) without human intervention. Thus, human annotators play a key role in the entire process of opinion mining, as they are responsible for interpretating and assigning an overall sentiment to each document. <br><br>

### DATASET
Hotel review data used in this research was collected from Opin Rank datasets (Ganesan and Zhai, 2012) comprising titled comments from guests about hotels in many cities worldwide. Only [customer reviews of hotels in London](https://user-images.githubusercontent.com/76821049/173886937-80017dff-d71f-4d3d-93e0-1614aedc7ced.png) were included in this study. <br>
<br>![image](https://user-images.githubusercontent.com/76821049/174006070-0ea5530f-d0f8-47b6-abe4-ab4825d41315.png)<br>
Due to the absence of sentiment labels, a human annotator was tasked with assigning sentiment labels to the reviews/documents by the emotional meaning of the keywords present in each document. Keywords were determined by selecting only matching words in the topic and body of text of each document. Careful consideration was given to the occurrence of word negation and sarcasm. Examples of such cases can be seen [here](https://user-images.githubusercontent.com/76821049/173890175-c10a0951-deea-45a7-a7a2-f7c53e4b75d4.png).  8,532 instances were carefully selected from the labeled data with balanced representation from both sentiment classes as shown in the [table](https://user-images.githubusercontent.com/76821049/173890433-14d40124-377c-4d52-978c-456aebfe0cef.png) and [chart](https://user-images.githubusercontent.com/76821049/173890546-a18d50f5-d674-4982-b659-3560780d8f13.png). [Here](https://user-images.githubusercontent.com/76821049/173887463-56e1c396-7f28-4a85-82a7-8ab835d34619.png), you can find the labelled dataset. The Python code used to label the reviews dataset can be found [here](https://github.com/Beegie01/Sentiment-Analysis-of-Titled-Hotel-Reviews-with-Shallow-Gated-Recurrent-Units/blob/main/Opin_Rank_annotation_hotel_data.ipynb)<br>

<br><br>Details of this research project can be found [here](https://github.com/Beegie01/Sentiment-Analysis-of-Titled-Hotel-Reviews-with-Shallow-Gated-Recurrent-Units/blob/main/Project_report.pdf).
