from django.db import models
from .utils import get_filtered_image
from PIL import Image
import numpy as np
import cv2
import imutils
from io import BytesIO
from django.core.files.base import ContentFile
from reportlab.pdfgen import canvas
from django.core.mail import send_mail



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

    def GeneratePDF(lista):
        try:
            nome_pdf = "Detritos"
            pdf = canvas.Canvas('{}.pdf'.format(nome_pdf))
            x = 720
            for nome, idade in lista.items():
                x -= 20
                pdf.drawString(247, x, '{} : {}'.format(nome, idade))
            pdf.setTitle(nome_pdf)
            pdf.setFont("Helvetica-Oblique", 14)
            pdf.drawString(245, 750, 'Lista de Convidados')
            pdf.setFont("Helvetica-Bold", 12)
            pdf.drawString(245, 724, 'Nome e idade')
            pdf.save()
            print('{}.pdf criado com sucesso!'.format(nome_pdf))
        except:
            print('Erro ao gerar {}.pdf'.format(nome_pdf))

    lista = {'Rafaela': '19', 'Jose': '15', 'Maria': '22', 'Eduardo': '24'}



    def save(self, *args, **kwargs):
        # abre imagem
        pil_img = Image.open(self.image)

        # converte imagem para um array e faz o processamento
        cv_img = np.array(pil_img)
        img = get_filtered_image(cv_img, self.action)

        # converte novamente para uma imagem
        im_pil = Image.fromarray(img)

        # salva
        buffer = BytesIO()
        im_pil.save(buffer, format='png')
        image_png = buffer.getvalue()

        self.image.save(str(self.image), ContentFile(image_png), save=False)
        """
        send_mail(
            'Subject here',
            'Here is the message.',
            'lucas.pinheiro28silva@gmail.com',
            ['lucas.pinheiro-lima@hotmail.com'],
            fail_silently=False,
        )
        """
        super().save(*args, **kwargs)

    GeneratePDF(lista)




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

        #cap = cv2.VideoCapture(f"../media/files/{self.file.name}")
        if(self.file == '0'):
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(f"{self.file}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 940)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 780)

        ret, frame = cap.read()
        # Segura o último frame para verificar se houve movimento
        last_frame = None
        # Texto que irá exibir o número de objetos
        text = ""
        # Caixas que irão contar os objetos
        boxes = []
        boxes.append(Box((100, 200), (10, 80)))
        boxes.append(Box((300, 350), (10, 80)))
        while cap.isOpened():
            _, frame = cap.read()
            # Processamento dos frames
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except:
                break
            # Para minimizar os pequenos detalhes
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if last_frame is None or last_frame.shape != gray.shape:
                last_frame = gray
                continue
            # Pega a diferença entre o frame atual e o último
            delta_frame = cv2.absdiff(last_frame, gray)
            last_frame = gray
            # Threshold - Atribuição dos valores em pixels de acordo com o limite
            thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]

            thresh = cv2.dilate(thresh, None, iterations=2)
            # Retorna uma lista de objetos
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Conversão
            contours = imutils.grab_contours(contours)
            # Looping em todos os objetos
            for contour in contours:
                # Pula se a diferença for pequena (pode ser ajustado)
                if cv2.contourArea(contour) < 500:
                    continue
                # Destaca o objeto colocando as caixas
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                text = "Objetos:"
                # Passa por todas as caixas
                for box in boxes:
                    box.frame_countdown -= 1
                    if box.overlap((x, y), (x + w, y + h)):
                        if box.frame_countdown <= 0:
                            box.counter += 1
                        # The number might be adjusted, it is just set based on my settings
                        box.frame_countdown = 20
                    text += " (" + str(box.counter) + " ," + str(box.frame_countdown) + ")"
            # Insere o texto da quantidade de elementos
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            # Insere as caixas
            for box in boxes:
                cv2.rectangle(frame, box.start_point, box.end_point, (255, 255, 255), 2)
            # Abre a janela
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

