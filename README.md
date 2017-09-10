# Pattern Recognition Project -- Object Classification 
![look][pic1]  

## Package Summary   

- Maintainer status: maintained
- Maintainer: Sha Luo <luoshasha1992@gmail.com>
- Author: Sha Luo <luoshasha1992@gmail.com>
- License: Apache
- Bug / feature tracker: https://github.com/Abbyls/pattern_recognition_project/issues
- Source: git https://github.com/Abbyls/pattern_recognition_project (branch: master)   
- Language: C++, Qt, Matlab

## Description
This is my course(Pattern Recognition) project which received the highest scores among the class. The project uses different techniques to classify different objects from a given image. The project is a pratical way to get to know different knowledge as follows:
- two-class linear classification;
- three-class linear classification;
- non-linear classification(SVM with kernel function);     


And the classification methods are: 
(1) color threshold method 
(2) Linear Discriminant Analysis(LDA)
(3) Manual Linear Classification
(4) Support Vector Machine(SVM) 
(5) My own method

There are basically 3 experiments, 
- The first experiment operates on "Image 1" (see the figure above); This uses 2-class linear classification with methods (1)-(5);    
- The second experiment operates on "Image 2"(see the figure above); This uses 3-class linear classification with methods (1)-(5);      
- The third experiment operates on "Image 3" (see the figure above); This uses linear classification with method (3) and non-linear classification with method (4). It can classify any number of classes.      

Another seperate experiment is to compare the differences between solving the SVM problem in the original space and the dual space. The source code is located in [matlab][3] folder.     

## Recommended Operating Environment
1. Ubuntu 14.04;
2. Qt5.3
3. OpenCV 2.0
4. Eigen 3.0
> The project uses the following OpenCV libraries:opencv_core, opencv_highgui, opencv_imgproc, opencv_ml. Please make sure these libraries exist(normally there should be no problem).

## Compile and Run
1. Open QtCreator(the IDE of Qt);
2. Click 'File' -> 'Open File or Project...' and then choose the [pattern_recognition_101.pro][1] file in src/ folder
3. Configure the project's 'Debug' and 'Release' path to debug/ or release/, for example:
   ![config_path][pic2]
   Actually, as long as both the path of these two folders are in the root directory of this project, then it is fine. Otherwise, the project cannot run successfully since it cannot find the image files located in [img/][2] folder.
4. Click 'Run' to run this code. Have fun!

[1]: src/pattern_recognition_101.pro
[2]: img/
[3]: matlab/

[pic1]: look.png
[pic2]: config_path.png
