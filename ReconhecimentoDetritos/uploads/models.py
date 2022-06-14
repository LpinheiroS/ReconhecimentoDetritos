from django.db import models
from .utils import get_filtered_image
from PIL import Image
import numpy as np
import cv2
import imutils
from io import BytesIO
from django.core.files.base import ContentFile

# Create your models here.
ACTION_CHOICES = (
    ('CANNY', 'identificar'),
)




class Box:
    def __init__(self, start_point, width_height):
        self.start_point = start_point
        self.end_point = (start_point[0] + width_height[0], start_point[1] + width_height[1])
        self.counter = 0
        self.frame_countdown = 0

    def overlap(self, start_point, end_point):
        if self.start_point[0] >= end_point[0] or self.end_point[0] <= start_point[0] or \
                self.start_point[1] >= end_point[1] or self.end_point[1] <= start_point[1]:
            return False
        else:
            return True




class Upload(models.Model):
    image = models.ImageField(upload_to='images')
    action = models.CharField(max_length=50, choices=ACTION_CHOICES)
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.id)

    def save(self, *args, **kwargs):
        # open image
        pil_img = Image.open(self.image)

        # convert the image to array and do some processing
        cv_img = np.array(pil_img)
        img = get_filtered_image(cv_img, self.action)

        # convert back to pil image
        im_pil = Image.fromarray(img)

        # save
        buffer = BytesIO()
        im_pil.save(buffer, format='png')
        image_png = buffer.getvalue()

        self.image.save(str(self.image), ContentFile(image_png), save=False)

        super().save(*args, **kwargs)


class Video(models.Model):
    #file = models.FileField(upload_to='files')
    file = models.CharField(max_length=50)


    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.id)

    def save(self, *args, **kwargs):

        '''self.file.save(str(self.file), ContentFile(self.recipes), save=False)

        super().save(*args, **kwargs)'''

        # open image
        #cap = cv2.VideoCapture(f"../media/files/{self.file.name}")
        if(self.file == '0'):
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(f"{self.file}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 940)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 780)

        # ret, frame = cap.read()
        # We will keep the last frame in order to see if there has been any movement
        last_frame = None
        # To build a text string with counting status
        text = ""
        # The boxes we want to count moving objects in
        boxes = []
        boxes.append(Box((100, 200), (10, 80)))
        boxes.append(Box((300, 350), (10, 80)))
        while cap.isOpened():
            _, frame = cap.read()
            # Processing of frames are done in gray
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except:
                break
            # We blur it to minimize reaction to small details
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # Need to check if we have a lasqt_frame, if not get it
            if last_frame is None or last_frame.shape != gray.shape:
                last_frame = gray
                continue
            # Get the difference from last_frame
            delta_frame = cv2.absdiff(last_frame, gray)
            last_frame = gray
            # Have some threshold on what is enough movement
            thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
            # This dilates with two iterations
            thresh = cv2.dilate(thresh, None, iterations=2)
            # Returns a list of objects
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Converts it
            contours = imutils.grab_contours(contours)
            # Loops over all objects found
            for contour in contours:
                # Skip if contour is small (can be adjusted)
                if cv2.contourArea(contour) < 500:
                    continue
                # Get's a bounding box and puts it on the frame
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # The text string we will build up
                text = "Objetos:"
                # Go through all the boxes
                for box in boxes:
                    box.frame_countdown -= 1
                    if box.overlap((x, y), (x + w, y + h)):
                        if box.frame_countdown <= 0:
                            box.counter += 1
                        # The number might be adjusted, it is just set based on my settings
                        box.frame_countdown = 20
                    text += " (" + str(box.counter) + " ," + str(box.frame_countdown) + ")"
            # Set the text string we build up
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            # Let's also insert the boxes
            for box in boxes:
                cv2.rectangle(frame, box.start_point, box.end_point, (255, 255, 255), 2)
            # Let's show the frame in our window
            cv2.imshow("Identificacao de objetos", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()







        """pil_img = Image.open(self.image)

        # convert the image to array and do some processing
        cv_img = np.array(pil_img)
        img = get_filtered_image(cv_img, self.action)

        # convert back to pil image
        im_pil = Image.fromarray(img)

        # save
        buffer = BytesIO()
        im_pil.save(buffer, format='png')
        image_png = buffer.getvalue()

        self.image.save(str(self.image), ContentFile(image_png), save=False)

        super().save(*args, **kwargs)"""

