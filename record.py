import face_recognition

# Load the known image and get the encoding
picture_of_me = face_recognition.load_image_file("biden.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# Load the unknown image and get the encoding
unknown_picture = face_recognition.load_image_file("shelile.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# Set a stricter tolerance
tolerance = 0.5

# Compare faces
results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding, tolerance=tolerance)

# Calculate the face distance
face_distances = face_recognition.face_distance([my_face_encoding], unknown_face_encoding)

if results[0]:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")

# Print the face distance for further insight
print(f"Face distance: {face_distances[0]}")
