from flask import Flask
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
from camera import VideoCamera
from datetime import datetime
from datetime import date
import datetime
import random
from random import seed
from random import randint
import cv2
import numpy as np
import threading
import os
import time
import shutil
import imagehash
import PIL.Image
from PIL import Image
from PIL import ImageTk
import urllib.request
import urllib.parse
from urllib.request import urlopen
import webbrowser
import argparse
import mysql.connector
import pyttsx3

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="autism"
)


app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
@app.route('/',methods=['POST','GET'])
def index():
    cnt=0
    act=""
    msg=""
    ff=open("det.txt","w")
    ff.write("1")
    ff.close()

    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()

    ff11=open("img.txt","w")
    ff11.write("1")
    ff11.close()

    ff=open("person.txt","w")
    ff.write("")
    ff.close()
    
    if request.method == 'POST':
        name = request.form['name']
        ff=open("name.txt","w")
        ff.write(name)
        ff.close()
        return redirect(url_for('index1')) 

    return render_template('index.html',msg=msg,act=act)


@app.route('/index1',methods=['POST','GET'])
def index1():
    cnt=0
    act=""
    msg=""


    return render_template('index1.html',msg=msg,act=act)




@app.route('/login1', methods=['POST','GET'])
def login1():
    result=""
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    if request.method == 'POST':
        username1 = request.form['uname']
        password1 = request.form['pass']
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM admin where username=%s && password=%s",(username1,password1))
        myresult = mycursor.fetchone()[0]
        if myresult>0:
            result=" Your Logged in sucessfully**"
            return redirect(url_for('admin')) 
        else:
            result="Your logged in fail!!!"
                
    
    return render_template('login1.html',result=result)



@app.route('/admin',methods=['POST','GET'])
def admin():
    act=request.args.get("act")
    mycursor = mydb.cursor()

    ff=open("myclass.txt","r")
    ojj=ff.read()
    ff.close()

    jdata=ojj.split(",")
    
    if request.method == 'POST':
        
        obj_name = request.form['obj_name']
        obj_uses = request.form['obj_uses']

        mycursor.execute("SELECT max(id)+1 FROM object_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        sql = "INSERT INTO object_data(id, obj_name,obj_uses) VALUES (%s, %s, %s)"
        val = (maxid, obj_name, obj_uses)
        
        mycursor.execute(sql, val)
        mydb.commit()
        

    mycursor.execute("SELECT * FROM object_data")
    value = mycursor.fetchall()

    ###
    if act=="del":
        did=request.args.get("did")

        mycursor.execute("delete from object_data where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('admin')) 
    ###
        
    return render_template('admin.html',value=value,jdata=jdata)


@app.route('/edit_obj',methods=['POST','GET'])
def edit_obj():
    act=request.args.get("act")
    vid=request.args.get("vid")
    mycursor = mydb.cursor()

    ff=open("myclass.txt","r")
    ojj=ff.read()
    ff.close()

    jdata=ojj.split(",")
    
    if request.method == 'POST':
        obj_uses = request.form['obj_uses']
        mycursor.execute("update object_data set obj_uses=%s where id=%s",(obj_uses,vid))
        mydb.commit()
        return redirect(url_for('admin')) 

    mycursor.execute("SELECT * FROM object_data where id=%s",(vid,))
    value = mycursor.fetchone()

    return render_template('edit_obj.html',value=value,jdata=jdata)

@app.route('/edit',methods=['POST','GET'])
def edit():
    act=request.args.get("act")
    vid=request.args.get("vid")
    mycursor = mydb.cursor()

    ff=open("myclass.txt","r")
    ojj=ff.read()
    ff.close()

    jdata=ojj.split(",")
    
    if request.method == 'POST':
        obj_name = request.form['obj_name']
        mycursor.execute("update test_image set answer=%s where id=%s",(obj_name,vid))
        mydb.commit()
        return redirect(url_for('add_image')) 

    mycursor.execute("SELECT * FROM test_image where id=%s",(vid,))
    value = mycursor.fetchone()

    return render_template('edit.html',value=value,jdata=jdata)


@app.route('/admin1',methods=['POST','GET'])
def admin1():
    act=request.args.get("act")
    mycursor = mydb.cursor()
    if request.method == 'POST':
        
        detail = request.form['detail']

        mycursor.execute("SELECT max(id)+1 FROM train_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        sql = "INSERT INTO train_data(id, utype, detail,fimg) VALUES (%s, %s, %s, %s)"
        val = (maxid, '', detail, '')
        print(sql)
        mycursor.execute(sql, val)
        mydb.commit()
        return redirect(url_for('add_photo',vid=maxid)) 

    mycursor.execute("SELECT * FROM train_data")
    value = mycursor.fetchall()

    ###
    if act=="del":
        did=request.args.get("did")

        mycursor.execute("SELECT count(*) FROM vt_face where vid=%s",(did,))
        cn = mycursor.fetchone()[0]
        if cn>0:
            mycursor.execute("SELECT * FROM vt_face where vid=%s",(did,))
            dd = mycursor.fetchall()
            for ds in dd:
                os.remove("static/frame/"+ds[2])

            mycursor.execute("delete from vt_face where vid=%s",(did,))
            mydb.commit()
                
        
        mycursor.execute("delete from train_data where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('admin1')) 
    ###
        
    return render_template('admin1.html',value=value)

def speak(audio):
    engine = pyttsx3.init()
    engine.say(audio)
    engine.runAndWait()
    
@app.route('/add_image',methods=['POST','GET'])
def add_image():
    act=request.args.get("act")
    mycursor = mydb.cursor()
    if request.method == 'POST':
        
        answer = request.form['answer']

        mycursor.execute("SELECT max(id)+1 FROM test_image")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        file = request.files['file']
        filename=file.filename
        fn="T"+str(maxid)+filename
        file.save(os.path.join("static/upload", fn))

            
        sql = "INSERT INTO test_image(id, test_image,answer) VALUES (%s, %s, %s)"
        val = (maxid, fn, answer)
        print(sql)
        mycursor.execute(sql, val)
        mydb.commit()
        return redirect(url_for('add_image',vid=maxid)) 

    mycursor.execute("SELECT * FROM test_image")
    value = mycursor.fetchall()

    ###
    if act=="del":
        did=request.args.get("vid")
        mycursor.execute("delete from test_image where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_image')) 
    ###
        
    return render_template('add_image.html',value=value)

@app.route('/add_test',methods=['POST','GET'])
def add_test():
    msg=""
    act=request.args.get("act")
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM admin where username='admin'")
    value = mycursor.fetchone()

    
    if request.method == 'POST':
        
        num_image = request.form['num_image']
        mycursor.execute("update admin set num_image=%s where username='admin'",(num_image,))
        mydb.commit()
        msg="ok"
        

    return render_template('add_test.html',msg=msg,value=value)

#Hidden Markov Model
def HMM():
    self.transmission_prob = transmission_prob
    self.emission_prob = emission_prob
    self.n = self.emission_prob.shape[1]
    self.m = self.emission_prob.shape[0]
    self.observations = None
    self.forward = []
    self.backward = []
    self.psi = []
    self.obs = obs
    self.emiss_ref = {}
    self.forward_final = [0 , 0]
    self.backward_final = [0 , 0]
    self.state_probs = []
    if obs is None and self.observations is not None:
        self.obs = self.assume_obs()

    def assume_obs(self):
        '''
        If observation labels are not given, will assume that the emission
        probabilities are in alpha-numerical order.
        '''
        obs = list(set(list(self.observations)))
        obs.sort()
        for i in range(len(obs)):
            self.emiss_ref[obs[i]] = i
        return obs

    def train(self, observations, iterations = 10, verbose=True):
        '''
        Trains the model parameters according to the observation sequence.

        Input:
        - observations: 1-D string array of T observations
        '''
        self.observations = observations
        self.obs = self.assume_obs()
        self.psi = [[[0.0] * (len(self.observations)-1) for i in range(self.n)] for i in range(self.n)]
        self.gamma = [[0.0] * (len(self.observations)) for i in range(self.n)]
        for i in range(iterations):
            old_transmission = self.transmission_prob.copy()
            old_emission = self.emission_prob.copy()
            if verbose:
                print("Iteration: {}".format(i + 1))
            self.expectation()
            self.maximization()

    def expectation(self):
        '''
        Executes expectation step.
        '''
        self.forward = self.forward_recurse(len(self.observations))
        self.backward = self.backward_recurse(0)
        self.get_gamma()
        self.get_psi()

    def get_gamma(self):
        '''
        Calculates the gamma matrix.
        '''
        self.gamma = [[0, 0] for i in range(len(self.observations))]
        for i in range(len(self.observations)):
            self.gamma[i][0] = (float(self.forward[0][i] * self.backward[0][i]) /
                                float(self.forward[0][i] * self.backward[0][i] +
                                self.forward[1][i] * self.backward[1][i]))
            self.gamma[i][1] = (float(self.forward[1][i] * self.backward[1][i]) /
                                float(self.forward[0][i] * self.backward[0][i] +
                                self.forward[1][i] * self.backward[1][i]))

    def get_psi(self):
        '''
        Runs the psi calculation.
        '''
        for t in range(1, len(self.observations)):
            for j in range(self.n):
                for i in range(self.n):
                    self.psi[i][j][t-1] = self.calculate_psi(t, i, j)

    def calculate_psi(self, t, i, j):
        '''
        Calculates the psi for a transition from i->j for t > 0.
        '''
        alpha_tminus1_i = self.forward[i][t-1]
        a_i_j = self.transmission_prob[j+1][i+1]
        beta_t_j = self.backward[j][t]
        observation = self.observations[t]
        b_j = self.emission_prob[self.emiss_ref[observation]][j]
        denom = float(self.forward[0][i] * self.backward[0][i] + self.forward[1][i] * self.backward[1][i])
        return (alpha_tminus1_i * a_i_j * beta_t_j * b_j) / denom

    def maximization(self):
        '''
        Executes maximization step.
        '''
        self.get_state_probs()
        for i in range(self.n):
            self.transmission_prob[i+1][0] = self.gamma[0][i]
            self.transmission_prob[-1][i+1] = self.gamma[-1][i] / self.state_probs[i]
            for j in range(self.n):
                self.transmission_prob[j+1][i+1] = self.estimate_transmission(i, j)
            for obs in range(self.m):
                self.emission_prob[obs][i] = self.estimate_emission(i, obs)

    def get_state_probs(self):
        '''
        Calculates total probability of a given state.
        '''
        self.state_probs = [0] * self.n
        for state in range(self.n):
            summ = 0
            for row in self.gamma:
                summ += row[state]
            self.state_probs[state] = summ

    def estimate_transmission(self, i, j):
        '''
        Estimates transmission probabilities from i to j.
        '''
        return sum(self.psi[i][j]) / self.state_probs[i]

    def estimate_emission(self, j, observation):
        '''
        Estimate emission probability for an observation from state j.
        '''
        observation = self.obs[observation]
        ts = [i for i in range(len(self.observations)) if self.observations[i] == observation]
        for i in range(len(ts)):
            ts[i] = self.gamma[ts[i]][j]
        return sum(ts) / self.state_probs[j]

    def backward_recurse(self, index):
        '''
        Runs the backward recursion.
        '''
        # Initialization at T
        if index == (len(self.observations) - 1):
            backward = [[0.0] * (len(self.observations)) for i in range(self.n)]
            for state in range(self.n):
                backward[state][index] = self.backward_initial(state)
            return backward
        # Recursion for T --> 0
        else:
            backward = self.backward_recurse(index+1)
            for state in range(self.n):
                if index >= 0:
                    backward[state][index] = self.backward_probability(index, backward, state)
                if index == 0:
                    self.backward_final[state] = self.backward_probability(index, backward, 0, final=True)
            return backward

    def backward_initial(self, state):
        '''
        Initialization of backward probabilities.
        '''
        return self.transmission_prob[self.n + 1][state + 1]

    def backward_probability(self, index, backward, state, final=False):
        '''
        Calculates the backward probability at index = t.
        '''
        p = [0] * self.n
        for j in range(self.n):
            observation = self.observations[index + 1]
            if not final:
                a = self.transmission_prob[j + 1][state + 1]
            else:
                a = self.transmission_prob[j + 1][0]
            b = self.emission_prob[self.emiss_ref[observation]][j]
            beta = backward[j][index + 1]
            p[j] = a * b * beta
        return sum(p)

    def forward_recurse(self, index):
        '''
        Executes forward recursion.
        '''
        # Initialization
        if index == 0:
            forward = [[0.0] * (len(self.observations)) for i in range(self.n)]
            for state in range(self.n):
                forward[state][index] = self.forward_initial(self.observations[index], state)
            return forward
        # Recursion
        else:
            forward = self.forward_recurse(index-1)
            for state in range(self.n):
                if index != len(self.observations):
                    forward[state][index] = self.forward_probability(index, forward, state)
                else:
                    # Termination
                    self.forward_final[state] = self.forward_probability(index, forward, state, final=True)
            return forward

    def forward_initial(self, observation, state):
        '''
        Calculates initial forward probabilities.
        '''
        self.transmission_prob[state + 1][0]
        self.emission_prob[self.emiss_ref[observation]][state]
        return self.transmission_prob[state + 1][0] * self.emission_prob[self.emiss_ref[observation]][state]

    def forward_probability(self, index, forward, state, final=False):
        '''
        Calculates the alpha for index = t.
        '''
        p = [0] * self.n
        for prev_state in range(self.n):
            if not final:
                # Recursion
                obs_index = self.emiss_ref[self.observations[index]]
                p[prev_state] = forward[prev_state][index-1] * self.transmission_prob[state + 1][prev_state + 1] * self.emission_prob[obs_index][state]
            else:
                # Termination
                p[prev_state] = forward[prev_state][index-1] * self.transmission_prob[self.n][prev_state + 1]
        return sum(p)

    def likelihood(self, new_observations):
        '''
        Returns the probability of a observation sequence based on current model
        parameters.
        '''
        new_hmm = HMM(self.transmission_prob, self.emission_prob)
        new_hmm.observations = new_observations
        new_hmm.obs = new_hmm.assume_obs()
        forward = new_hmm.forward_recurse(len(new_observations))
        return sum(new_hmm.forward_final)


    model = HMM(transmission, emission)
    model.train(observations)
    print("Model transmission probabilities:\n{}".format(model.transmission_prob))
    print("Model emission probabilities:\n{}".format(model.emission_prob))

###

@app.route('/test_img',methods=['POST','GET'])
def test_img():
    msg=""
    msg_input=""
    act=request.args.get("act")
    qm=request.args.get("qm")
    value1=[]
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM admin where username='admin'")
    value = mycursor.fetchone()
    num_image=int(value[2])

    mycursor.execute("delete from au_temp")
    mydb.commit()

    mycursor.execute("SELECT * FROM test_image order by rand() limit 0,%s",(num_image,))
    dvalue = mycursor.fetchall()
    for value2 in dvalue:
        
        mycursor.execute("SELECT max(id)+1 FROM au_temp")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        sql = "INSERT INTO au_temp(id, img_id,answer,uans) VALUES (%s, %s, %s,%s)"
        val = (maxid,value2[0],'','')
        mycursor.execute(sql, val)
        mydb.commit()


    return render_template('test_img.html',msg=msg,act=act,qm=qm,value1=value1)


@app.route('/test_img1',methods=['POST','GET'])
def test_img1():
    msg=""
    msg_input=""
    st=""
    stt=""
    mess=""
    v1=0
    v2=0
    s1=""
    score=0
    act=request.args.get("act")
    qm=request.args.get("qm")
    value1=[]
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM admin where username='admin'")
    value = mycursor.fetchone()
    num_image=int(value[2])
    
    if request.method == 'POST':
        msg_input = request.form['res']

    if act=="1":
        s=1

        qm1=int(qm)

        if qm1<num_image:

            mycursor.execute("SELECT * FROM au_temp limit %s,1",(qm1,))
            value11 = mycursor.fetchone()
            qid=value11[1]

            
            mycursor.execute("SELECT * FROM test_image where id=%s",(qid,))
            value1 = mycursor.fetchone()
            qid=value1[0]
            ans=value1[2]
            qm1=qm1+1
            qm=str(qm1)
            print("qm")
            print(qm1)
            
            if msg_input=="":
                s=1
            else:
                mg=msg_input.lower()
                if '.' in mg:
                    mg1=mg.split('.')
                    mg=mg1[0]
                    
                print(mg)
                if ans==mg:
                    print("correct")
                    mycursor.execute("update au_temp set mark=1,status=1 where img_id=%s",(qid,))
                    mydb.commit()
                    mess="Correct Answer"
                    st="1"
                    stt="1"
                else:
                    stt="2"
                    mycursor.execute("update au_temp set status=1 where img_id=%s",(qid,))
                    mydb.commit()
                    mess="Wrong answer! This is "+ans
                    st="1"
                    print("no")
                    speak(mess)

        else:
            act="2"
            mycursor.execute("SELECT count(mark) FROM au_temp where mark=1")
            vv = mycursor.fetchone()[0]

            v1=vv
            v2=num_image-vv
            if vv>0:
                rr=(vv/num_image)*100
                score=round(rr,2)
            else:
                score=0

            print(v1)
            print(v2)
            print(score)
            if score>=90:
                s1="1"
            elif score>=70:
                s1="2"
            elif score>=50:
                s1="3"
            else:
                s1="4"


    elif act=="2":
    
        act=2

        

    else:
        s=1
        #mycursor.execute("delete from au_temp")
        #mydb.commit()

        '''mycursor.execute("SELECT * FROM test_image order by rand() limit 0,%s",(num_image,))
        dvalue = mycursor.fetchall()
        for value2 in dvalue:
            
            mycursor.execute("SELECT max(id)+1 FROM au_temp")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            
            sql = "INSERT INTO au_temp(id, img_id,answer,uans) VALUES (%s, %s, %s,%s)"
            val = (maxid,value2[0],'','')
            mycursor.execute(sql, val)
            mydb.commit()'''


    return render_template('test_img1.html',msg=msg,act=act,qm=qm,value1=value1,st=st,v1=v1,v2=v2,score=score,stt=stt,mess=mess,s1=s1)


@app.route('/page1',methods=['POST','GET'])
def page1():

    return render_template('page1.html')




@app.route('/add_photo',methods=['POST','GET'])
def add_photo():
    vid=""
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()

    #ff2=open("mask.txt","w")
    #ff2.write("face")
    #ff2.close()
    act = request.args.get('act')
    
    if request.method=='GET':
        vid = request.args.get('vid')
        ff=open("user.txt","w")
        ff.write(str(vid))
        ff.close()

    cursor = mydb.cursor()
    
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+vid+".jpg"
        

        cursor.execute('delete from vt_face WHERE vid = %s', (vid, ))
        mydb.commit()

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vface1=vid+"_"+str(v1)+".jpg"
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM vt_face")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface=vid+"_"+str(i)+".jpg"
            sql = "INSERT INTO vt_face(id, vid, vface) VALUES (%s, %s, %s)"
            val = (maxid, vid, vface)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
            
        cursor.execute('update train_data set fimg=%s WHERE id = %s', (vface1, vid))
        mydb.commit()
        shutil.copy('static/faces/f1.jpg', 'static/photo/'+vface1)
        return redirect(url_for('view_photo',vid=vid,act='success'))
        
    
    cursor.execute("SELECT * FROM train_data")
    data = cursor.fetchall()
    return render_template('add_photo.html',data=data, vid=vid)

@app.route('/add_photo1',methods=['POST','GET'])
def add_photo1():
    vid=""
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()

    ff2=open("mask.txt","w")
    ff2.write("mask")
    ff2.close()
                
    if request.method=='GET':
        vid = request.args.get('vid')
        ff=open("user.txt","w")
        ff.write(vid)
        ff.close()
    
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+vid+".jpg"
        cursor = mydb.cursor()

        cursor.execute('delete from vt_face WHERE vid = %s && mask_st=1', (vid, ))
        mydb.commit()

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vface1="m"+vid+"_"+str(v1)+".jpg"
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM vt_face")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface="m"+vid+"_"+str(i)+".jpg"
            sql = "INSERT INTO vt_face(id, vid, vface, mask_st) VALUES (%s, %s, %s, %s)"
            val = (maxid, vid, vface, '1')
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
            
        cursor.execute('update train_data set fimg=%s WHERE id = %s', (vface1, vid))
        mydb.commit()
        shutil.copy('faces/f1.jpg', 'static/photo/'+vface1)
        return redirect(url_for('view_cus',vid=vid,act='success'))
        
    cursor = mydb.cursor()
    cursor.execute("SELECT * FROM train_data")
    data = cursor.fetchall()
    return render_template('add_photo1.html',data=data, vid=vid)

@app.route('/view_cus',methods=['POST','GET'])
def view_cus():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register")
    value = mycursor.fetchall()
    return render_template('view_cus.html', result=value)

def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))


###Preprocessing
@app.route('/view_photo',methods=['POST','GET'])
def view_photo():
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()

    if request.method=='POST':
        print("Training")
        vid=request.form['vid']
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        dt = cursor.fetchall()
        for rs in dt:
            ##Preprocess
            path="static/frame/"+rs[2]
            path2="static/process1/"+rs[2]
            mm2 = PIL.Image.open(path).convert('L')
            rz = mm2.resize((200,200), PIL.Image.ANTIALIAS)
            rz.save(path2)
            
            '''img = cv2.imread(path2) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            path3="static/process2/"+rs[2]
            cv2.imwrite(path3, dst)'''
            #noice
            img = cv2.imread('static/process1/'+rs[2]) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            fname2='ns_'+rs[2]
            cv2.imwrite("static/process1/"+fname2, dst)
            ######
            ##bin
            image = cv2.imread('static/process1/'+rs[2])
            original = image.copy()
            kmeans = kmeans_color_quantization(image, clusters=4)

            # Convert to grayscale, Gaussian blur, adaptive threshold
            gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

            # Draw largest enclosing circle onto a mask
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                ((x, y), r) = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
                break
            
            # Bitwise-and for result
            result = cv2.bitwise_and(original, original, mask=mask)
            result[mask==0] = (0,0,0)

            
            ###cv2.imshow('thresh', thresh)
            ###cv2.imshow('result', result)
            ###cv2.imshow('mask', mask)
            ###cv2.imshow('kmeans', kmeans)
            ###cv2.imshow('image', image)
            ###cv2.waitKey()

            cv2.imwrite("static/process1/bin_"+rs[2], thresh)
            

            ###RPN - Segment
            img = cv2.imread('static/process1/'+rs[2])
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/fg_"+rs[2]
            segment.save(path3)
            ####
            img = cv2.imread('static/process2/fg_'+rs[2])
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/fg_"+rs[2]
            segment.save(path3)
            '''
            img = cv2.imread(path2)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/"+rs[2]
            segment.save(path3)
            '''
            #####
            image = cv2.imread(path2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 50, 100)
            image = Image.fromarray(image)
            edged = Image.fromarray(edged)
            path4="static/process3/"+rs[2]
            edged.save(path4)
            ##
            #shutil.copy('static/img/11.png', 'static/process4/'+rs[2])
       
        return redirect(url_for('view_photo1',vid=vid))
        
    return render_template('view_photo.html', result=value,vid=vid)

###
#CNN
def CNN():
    #Lets start by loading the Cifar10 data
    (X, y), (X_test, y_test) = cifar10.load_data()

    #Keep in mind the images are in RGB
    #So we can normalise the data by diving by 255
    #The data is in integers therefore we need to convert them to float first
    X, X_test = X.astype('float32')/255.0, X_test.astype('float32')/255.0

    #Then we convert the y values into one-hot vectors
    #The cifar10 has only 10 classes, thats is why we specify a one-hot
    #vector of width/class 10
    y, y_test = u.to_categorical(y, 10), u.to_categorical(y_test, 10)

    #Now we can go ahead and create our Convolution model
    model = Sequential()
    #We want to output 32 features maps. The kernel size is going to be
    #3x3 and we specify our input shape to be 32x32 with 3 channels
    #Padding=same means we want the same dimensional output as input
    #activation specifies the activation function
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
                     activation='relu'))
    #20% of the nodes are set to 0
    model.add(Dropout(0.2))
    #now we add another convolution layer, again with a 3x3 kernel
    #This time our padding=valid this means that the output dimension can
    #take any form
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
    #maxpool with a kernet of 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #In a convolution NN, we neet to flatten our data before we can
    #input it into the ouput/dense layer
    model.add(Flatten())
    #Dense layer with 512 hidden units
    model.add(Dense(512, activation='relu'))
    #this time we set 30% of the nodes to 0 to minimize overfitting
    model.add(Dropout(0.3))
    #Finally the output dense layer with 10 hidden units corresponding to
    #our 10 classe
    model.add(Dense(10, activation='softmax'))
    #Few simple configurations
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])
    #Run the algorithm!
    model.fit(X, y, validation_data=(X_test, y_test), epochs=25,
              batch_size=512)
    #Save the weights to use for later
    model.save_weights("cifar10.hdf5")
    #Finally print the accuracy of our model!
    print("Accuracy: &2.f%%" %(model.evaluate(X_test, y_test)[1]*100))



@app.route('/view_photo1',methods=['POST','GET'])
def view_photo1():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo1.html', result=value,vid=vid)

@app.route('/view_photo11',methods=['POST','GET'])
def view_photo11():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo11.html', result=value,vid=vid)

@app.route('/view_photo2',methods=['POST','GET'])
def view_photo2():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo2.html', result=value,vid=vid)    

@app.route('/view_photo3',methods=['POST','GET'])
def view_photo3():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo3.html', result=value,vid=vid)

@app.route('/view_photo4',methods=['POST','GET'])
def view_photo4():
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM vt_face where vid=%s",(vid, ))
        value = mycursor.fetchall()
    return render_template('view_photo4.html', result=value,vid=vid)

@app.route('/message',methods=['POST','GET'])
def message():
    vid=""
    name=""
    if request.method=='GET':
        vid = request.args.get('vid')
        mycursor = mydb.cursor()
        mycursor.execute("SELECT name FROM register where id=%s",(vid, ))
        name = mycursor.fetchone()[0]
    return render_template('message.html',vid=vid,name=name)




@app.route('/process',methods=['POST','GET'])
def process():
    msg=""
    ss=""
    uname=""
    mess=""
    act=""
    det=""
    # (0, 1) is N
    SCALE = 2.2666 # the scale is chosen to be 1 m = 2.266666666 pixels
    MIN_LENGTH = 150 # pixels

    if request.method=='GET':
        act = request.args.get('act')
        
    ff3=open("img.txt","r")
    mcnt=ff3.read()
    ff3.close()

    cursor = mydb.cursor()

    '''try:

        mcnt1=int(mcnt)
        print(mcnt1)
        if mcnt1>=2:
        
            cutoff=8
            act="1"
            cursor.execute('SELECT * FROM vt_face')
            dt = cursor.fetchall()
            for rr in dt:
                hash0 = imagehash.average_hash(Image.open("static/frame/"+rr[2])) 
                hash1 = imagehash.average_hash(Image.open("static/faces/f1.jpg"))
                cc1=hash0 - hash1
                print("cc="+str(cc1))
                if cc1<=cutoff:
                    vid=rr[1]
                    cursor.execute('SELECT * FROM train_data where id=%s',(vid,))
                    rw = cursor.fetchone()
                    
                    msg="Hai "+rw[2]
                    ff=open("person.txt","w")
                    ff.write(msg)
                    ff.close()
                    print(msg)
                 
                    break
                else:
                    msg="Unknown person found"
                    ff=open("person.txt","w")
                    ff.write(msg)
                    ff.close()
                
    except:
        print("excep")'''
        

    msg1=""
    msg2=""
    mess=""
    ff=open("get_value.txt","r")
    get_value=ff.read()
    ff.close()
    s=""
    if get_value=="":
        s="1"
    else:
        
        msg1="Object Name: "+get_value
        cursor.execute('SELECT count(*) FROM object_data where obj_name=%s',(get_value,))
        cn = cursor.fetchone()[0]
        if cn>0:
            cursor.execute('SELECT * FROM object_data where obj_name=%s',(get_value,))
            dtt = cursor.fetchone()
            msg2=dtt[2]
    
        
        mess=msg1+" "+msg2
        speak(mess)

    return render_template('process.html',msg1=msg1,msg2=msg2,mess=mess,act=act)



@app.route('/clear_data',methods=['POST','GET'])
def clear_data():
    ff=open("person.txt","w")
    ff.write("")
    ff.close()

    ff1=open("get_value.txt","w")
    ff1.write("")
    ff1.close()
    return render_template('clear_data.html')

@app.route('/user_view')
def user_view():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register")
    result = mycursor.fetchall()
    return render_template('user_view.html', result=result)



@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    #session.pop('username', None)
    return redirect(url_for('index'))

def gen(camera):
    
    while True:
        frame = camera.get_frame()
        
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
        

def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
