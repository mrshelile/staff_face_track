from flask import Flask, Response
import face_recognition
import numpy as np
import requests
import json
import ujson
from io import BytesIO
from PIL import Image
import datetime
import cv2  # Import OpenCV for webcam access
import timeit

link ="http://localhost:1337"
latency = 15

def isTodayAttended(employee_number) :
    res = requests.get(link+f"/api/attendances?populate=*")
    data = json.loads(res.content)['data'] 
    if not data:
        return -1
    for item in data:
        emp_num = item['attributes']['employee']['data']['attributes']['employee_number']
        entry = item['attributes']['entry']
        date_format = "%Y-%m-%dT%H:%M:%S.%fZ"

        # Parse the string to datetime
        parsed_date = datetime.datetime.strptime(entry, date_format)
        
        now = datetime.datetime.now()

        # checking is today attended the create exit time
        if(parsed_date.year == now.year and 
           parsed_date.month == now.month  and parsed_date.day == now.day):
            
            if str(emp_num)==str(employee_number):
                
                return item['id']
        
    
    return -1

def exitUpdate(id):
    current_datetime = datetime.datetime.now()
    current_datetime_str = current_datetime.isoformat()
    attendance ={
            "data":
            {
                "exit":current_datetime_str,
                # "isRead":ujson.dumps(datetime.datetime.now())
            }
    }
    res = requests.put(link+f"/api/attendances/{id}",json=attendance)
    return res.status_code

def isPersonAvailable(employee_number):
    
    res = requests.get(link+f"/api/employees?filters[employee_number][$eq]={employee_number}")
    data = json.loads(res.content)['data']
    # print(data[0]['id'])
    # exit()
    if data:
        
        attended_id =isTodayAttended(employee_number)
        # print(attended_id)
        if(attended_id!=-1):
            return exitUpdate(attended_id)
        else:
            current_datetime = datetime.datetime.now()

            # Convert the datetime to a string
            current_datetime_str = current_datetime.isoformat()
            attendance ={
                "data":
                {
                    "employee":data[0]['id'],
                    "entry":current_datetime_str,
                    # "isRead":ujson.dumps(datetime.datetime.now())
                }
            }
            res = requests.post(link+"/api/attendances/",json=attendance)
            
            return res.status_code
    else:
        print("not found")
    return 500

def recordAttendance(data):
    # cap = cv2.VideoCapture(0)  # 0 for default webcam
    video_capture = cv2.VideoCapture(0)
    exec_time = None

   
    known_face_encodings =[image_encode["displays"][0] for image_encode in data]
        # print(item["displays"]) 
    # print(known_face_encodings)
    known_face_names = [image_encode["employee_number"] for image_encode in data]
    foundPeople = [False for _ in data]
    # print(known_face_names)
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read(0)

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            # rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame ,face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.5)
                # print(matches)
                name = "Unknown"
                # print(matches)
                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    
                    if not foundPeople[first_match_index]:
                        status =isPersonAvailable(name)   
                        exec_time = timeit.default_timer()
                        if status == 200:
                            foundPeople[first_match_index]=True
                    else:
                        t_1 = timeit.default_timer()
                        elapsed_time = round((t_1 - exec_time) * 10 ** 6, 3)
                        elapsed_time_seconds = elapsed_time / 10**6
                        print(elapsed_time_seconds)
                        if elapsed_time_seconds> latency:
                            status =isPersonAvailable(name)
                
                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (5, 255, 255),  2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (5, 255, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 0, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        # ret, buffer = cv2.imencode('.jpg', frame)
        # frame = buffer.tobytes()
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
def fetchData():
    response_API =requests.get(link+"/api/employees?populate=*")
    jdata = json.loads(response_API.content)['data']
    # print(jdata)
    data=[]
    for item in jdata:
        image_array = None
        image_array1 = None
        image = None
        employee_number = item['attributes']['employee_number']
        # displays = [link+img['attributes']['url'] for img in item['attributes']['picture']['data']]
      
        displays =link+ item['attributes']['picture']['data']['attributes']['url']
        response = requests.get(displays)
        image_data = response.content
        # Load the image into PIL
        image = Image.open(BytesIO(image_data))
        # Convert the PIL image to a numpy array
        image_array = np.array(image)
        # print("Image shape:", image_array.shape)\

        # print(image_array)
        # exit()
        image_array1 = face_recognition.face_encodings(image_array)
        # print("Number of face encodings:", len(image_array1))
        # print(image_array)
        data.append({"employee_number":employee_number,"displays":image_array1})
    return data


recordAttendance(fetchData())
# print(isPersonAvailable(1234))
# print(exitUpdate(5))
# fetchData()
