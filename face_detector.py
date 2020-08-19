from flask import Flask, render_template, request, redirect
import cv2
import numpy as np


app = Flask(__name__)


clf= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(img):
    global clf
    if(len(img.shape)==3):
    	img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
    	img_gray=img.copy()

    faces= clf.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces)!=0:
    	face_img=img.copy()

    	for face in faces:
    		x,y,w,h=face
    		cv2.rectangle(face_img, (x,y), (x+w,y+h), (0,255,0), 5)
    		return face_img
    else:
    	return img


def modify_save_video(filename):
    cap= cv2.VideoCapture('static\\images\\'+ filename)

    w= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer= cv2.VideoWriter('static\\images\\w.webm', cv2.VideoWriter_fourcc('V','P','8','0'), 25, (w,h))

    for i in range(100):
        ret, frame= cap.read()
        
        if ret==True:
            face_frame= detect_face(frame)
            writer.write(face_frame)
    #         cv2.imshow('frame', face_frame)
            if cv2.waitKey(7)==27:
                break
        else:
            break

    writer.release()
    cap.release()




@app.route("/")
@app.route("/Home")
def home():
	return render_template("face_detection.html", img=0, video=0)




@app.route("/Test", methods=["GET","POST"])
def test():

	if request.method=="POST":
		fig = request.files['pic']
		# print(type(fig))
			# file = request.files['file']
		fig.save('static\\images\\'+ fig.filename)
		img= cv2.imread('static\\images\\'+ fig.filename)
		face_img= detect_face(img)
		cv2.imwrite('static\\images\\new.jpg', face_img)

		return render_template("face_detection.html", img='new.jpg', video=0)


	return render_template("face_detection.html", img=0, video=0)










@app.route("/Video", methods=["GET","POST"])
def video():

	if request.method=="POST":
		# f=int(request.form["first"])
		v = request.files['vid']
	# 	# print(type(fig))
	# 		# file = request.files['file']
		v.save('static\\images\\'+ v.filename)
	# 	img= cv2.imread('C:\\Users\\GM SIR\\Desktop\\Face Detection\\static\\images\\'+ fig.filename)
	# 	face_img= detect_face(img)
	# 	cv2.imwrite('C:\\Users\\GM SIR\\Desktop\\Face Detection\\static\\images\\new.jpg', face_img)

		modify_save_video(v.filename)

		return render_template("face_detection.html", img=0, video=1)


	# return render_template('video_test.html')




if __name__ == "__main__":
    app.run(debug=True)



# app.run(debug=True)



















# import numpy
# import pandas
# import sklearn
# import pickle



# def classify(text):
# 	loaded_model= pickle.load( open( "model_nlp.pkl", "rb" ) )
# 	predict_this=loaded_model.predict([text])

# 	if predict_this[0]==0:
# 		return 'Negative Review'
# 	else:
# 		return 'Positive Review'

	# return predict_this



# print(classify('Absolutely loved the movie and its character so well directed. Enjoyed it thoroughly. Great work! hope such work continues.'))


# @app.route("/")
# @app.route("/Home" , methods=["GET","POST"])
# def home():
	
# 	if request.method=="POST":
# 		try:
# 			rev=str(request.form["user_review"])
# 			# a='good'
# 			return render_template("movie_review_nlp.html", ans=classify(rev))
# 		except:
# 			return "Something went wrong."

# 	return render_template("movie_review_nlp.html", ans='ok')




# if __name__ == "__main__":
#     app.run(debug=True)





