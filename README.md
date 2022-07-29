# Social-Media-Trends
## Made by Anjali Priya,Prakhar Nautiyal and Priya Kumari ##

![image](https://user-images.githubusercontent.com/88290919/181722681-9ddc264d-d7af-4e15-9552-6ea8ffcdbfbf.png)

### Problem Statement in Flipkart Grid 4.0 ###

Problem Definition: Extract Trends from social media data
As part of this challenge, teams are expected to identify trends from social media data; From all the
products available on Flipkart identify trending products, utilize all signals available (ex. posts, sessions,
check-ins, social graphs, media content, etc.). Output should also have photos, videos, gifs which can be
used on Flipkart app.Preferred tech: Open source
Bonus: Signal extraction from multiple social media channels (ex. FB, Instagram, Twitter, etc.)

### Key Vision ###

Producables -

1. A system for ranking products on e-commerce websites to identify the trends
2. Strategy to evaluate things that are popular and trailing on fashion portals and social media platforms
3. To make the solution scaleable

### Key sub-problems identified ###

1. Collecting information from e-commerce and fashion websites
2. Removing undesirable artefacts from image data
3. Discovering feature encodings for each image
4. In order to successfully combine the rating and number of reviews, a popularity metre (PM) must be calculated.
5. Using picture clustering to identify what is trending and what is lagging by grouping the photos based on their encodings

### Tech stack ###

1. Python 3.6
2. Selenium
3. Keras
4. TensorFlow
5. Matplotlib
6. Sklearn
7. Numpy
8. Pandas

## 1. Web Scraping the image data ##

* Following are the e-commerce websites and social media platform to extract data from
   1. Flipkart
   2. Vogue
   3. Amazon
   4. Myntra
   
* We gathered information on the product name, rating, number of reviews, and image from websites like Flipkart and Amazon.

* We gathered the fashion pictures from the various websites.

* This stage can be readily scaled up because the scripts are simple to adapt to function on other websites by altering a few variables in accordance with the website   architecture.

* The entire scraped data is transformed into a Pandas dataframe and then saved as a CSV.


## 2. Image downloads and Object Detection ##

1. Running the image download script.py will allow you to download the images from the image links included in the CSV.
2. A pretrained YOLOv3 architecture that was trained on the DeepFashion2 dataset was used for object detection.
3. Run the new image demo.py script to copy the code from this repository and utilise it.
4. This model recognises the  object categories and isolates the target inside the image's bounding box.
5. After saving the images, feature extraction is performed on it.

## 3.  Learning Feature Encodings ##

1. In order to represent our images for later processing, we needed a way to extract the features from each item
2. We trained a model using the keras library and tensorflow backend
3. Our model was based on the CNN architecture which is known in the Computer Vision world for being able to learn features from images
4. We recreated some of the images using the encodings we got and the results were very promising, indicating that out feature encodings/representations are accurate
5. To create the model, run the script encoder_training_script.py

## 4. Computing the Popularity Metric (PM) ##

1. We wanted consider both ratings and the number of ratings in our attempt to rank all the products effectively
2. We came up with a popularity measure which combines the two properly
![image](https://user-images.githubusercontent.com/88290919/181727083-b0186367-36c3-4438-887b-c6170a749927.png)
3. A Bayesian view of the beta distribution was adopted to come up with a formula to give us a PM(Popularity Metric) given the rating and number of ratings 
4. We loaded in all our e-Commerce data, calculated the feature encodings using the model mentioned earlier
5. Then computed the PM for each product
6. Then trained a model to predict the PM given a set of encodings - we can now compare the predicted performance of different products on e-Commerce sites, this is      especially useful for designers that want to know how the public would react to their clothes
7. To create and train the model, run pm_model_train_script.py
8. Once the model is created, you can run pm_predictor_script.py to predict the PM for any input image

## 5. Feature Clustering ##

1. We performed clustering on a user-selected group of photographs using the previously calculated encodings to show the leading and lagging products in the set of        images under consideration.
2. The Silhouette coefficient was used to test and compare 5 clustering techniques, and K means clustering produced the best outcomes.
3. The most prevalent/trending clothing styles were represented by the largest cluster, and the least prevalent ones by the smallest clusters.
4  Run clustering script.py to test this.

# Implementation Instructions #

To be added later...

![image](https://user-images.githubusercontent.com/88290919/181727962-dafd498d-f759-415b-9539-a6c23d69a785.png)

