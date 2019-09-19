# car-logo-recognition

Image classification using CNN, Keras, TensorFlow, Python, Flask.

Paste the contents in javaScriptConsole in your browser's console in three stages as in the file. This will generate a urls.txt file in your downloads folder.

Run datasetCrawl.py. This will download all the images from urls in urls.txt

I used the LeNet architecture for two classes using Binary Cross Entropy Loss only to realise that it yielded 70=80% accuracy. So, in order to improve the accuracy, I tried switching to a simple neural network with multiple hidden layers which yielded around 88% accuracy. As a result of the use of Binary Cross Entropy Loss, this worked on two classes only. 

I then switched to VGG16 model which took over 18 hours to train on my Dell laptop with NVIDIA GeForce 920MX graphic card and ended with an accuracy of 98%, using 5 classes: Audi, Honda, Hyundai, Volkswagen, Porsche. The prediction code for this model proved to be unfruitful as it kept giving incorrect predictions because I had a relatively small dataset. With a perfect dataset, this program should work just fine.

train_network.py and test_network.py uses lenet architecture

trainModelVGG.py and testModelVGG.py uses VGG architecture

index1.py is the RESTful POST API that allows you to select an image from your file explorer and gives the result as output.

*P.S.: I learned all of this from the internet*
