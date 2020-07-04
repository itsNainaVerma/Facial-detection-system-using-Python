# Facial-detection-system-using-Python
Project Summary-

This project describes the efficient algorithm that automatically registers attendance for the students attending the class by taking a photo of them using digital camera which continuously captures images of students and then comparing their faces with the ones stored in the database, if the system recognized any face that are stored in the database, the attendance is given for the recognized faces without needing any manual work to be done by the teacher present in the class. This is done by four stages. Firstly, the images of the student enrolled in the college are clicked and stored in a database for easy accessibility. Secondly, a camera is placed in the front of the class which detects the students attending the class, the output of which is sent to the third stage i.e. face recognition system which checks it with the students in the database and if the student face matches with one in the database attendance is given.

Index Terms- Python, Machine Learning, Deep Learning, CNN, Facial Recognition, OpenCV

PROPOSED WORK
The complete system is divided into 3 Modules i.e.

A. Database Creation:
Firstly, we need approximately 20-30 pictures of each student with different expressions so that the model trains in a better way and can classify faces efficiently. In order to create a quality rich database, we have to apply OpenCV functionalities to the data for better detection of faces.

B. Face Detection:
The Viola-Jones algorithm was implemented for Face detection. This algorithm consists of the cascading of Haar features and Adaboost.

C. Face Recognition:
Face recognition is done using Fisherfaces face recognizer. FisherFaces face recognizer algorithm extracts principal components that differentiate one person from the others. In that sense, an individual's components do not dominate (become more useful) over the others.
