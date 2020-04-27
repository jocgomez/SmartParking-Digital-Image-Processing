import cv2
import numpy as np
#import pygame
import  sys, os.path
import json
##from firebase import firebase
import gc
##from RPLCD import CharLCD
##import RPi.GPIO as GPIO
##import RPLCD as RPLCD
import time

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

def calcularNodo(img):
    entro = False
    
    #converting frame(img) from BGR (Blue-Green-Red) to HSV (hue-saturation-value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #defining the range of Yellow color
    yellow_lower = np.array([22,60,200],np.uint8)
    yellow_upper = np.array([60,255,255],np.uint8)
    
    #finding the range yellow colour in the image
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    #Morphological transformation, Dilation        
    kernal = np.ones((5 ,5), "uint8")
    blue=cv2.dilate(yellow, kernal)
    res=cv2.bitwise_and(img, img, mask = yellow)

    #Tracking Colour (Yellow) 
    __,contours,hierarchy = cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
              area = cv2.contourArea(contour)
              if(area>30):
                  entro = True

    return entro

def dibujarRuta(imgNodo,posInicial,estado1,estado2,estado3,estado4):

    #entro: Variable para saber si el auto con el pato se encuentra sobre un nodo
    #disponible: Variable para pintar el camino en caso de que haya un slot disponible
    #dibujar: Variable para pintar el camino 1 sola vez mientras se encuentra en el mismo nodo
    #anterior_path: Variable para limpiar el camino cuando cambie de nodo
    #path: la ruta en coordenadas x,y al slot mas cercano, inicia en 0
    
    entro = False
    disponible = False
    dibujar = True
    anterior_path = 0
    path = 0
    slot=0
    
    entro = calcularNodo(imgNodo)
#    pygame.display.update()
    if(entro):
        #Nodo inicio y nodo Fin(Parqueadero)
        start = posInicial

        if(estado4 == False):
            end = (3,4)
            disponible = True
            slot=4
        elif(estado3 == False):
            end = (3,3)
            disponible = True
            slot=3
        elif(estado2 == False):
            end = (3,2)
            disponible = True
            slot=2
        elif(estado1 == False):
            end = (3,1)
            disponible = True
            slot=1

        if(disponible):
            #El camino mas corto luego lo paso a matriz
            path = astar(maze, start, end)
            path_matrix = np.asmatrix(path)
##            path2=str(path)
##            print(path_matrix)

            ##ARREGLAR, anterior_path siempre es 0
            if(path != anterior_path):                
#                iniciarMapa()
                dibujar = True
##                print("prueba")
            else:
                dibujar = False

            anterior_path = path

            #Recorro el resultado del camino mas corto y busco esas coordenadas en el MAZE, y si los valores del maze
            #En esas coordenadas es 0, se pinta un recuadro verde
            for y in range(path_matrix.shape[0]):
                val= path_matrix[y]
                y = val[0,0]
                x = val[0,1]
#                if(maze[val[0,0]][val[0,1]] == 0):
#                    
#                    pygame.draw.rect(windowSurfaceObj, greenColor, (x*u, y*u, u, u))
#                    pygame.display.update()
    return path,slot

#def iniciarMapa():
#    
#    windowSurfaceObj.fill(whiteColor)    
#
#    #Recorro MAZE y donde halla un 1 pongo un cuadro negro PAREDES
#    for x in range(0, mazeHeight):
#        for y in range(0, mazeWidth):
#            if(maze[x][y] == 1):
#                pygame.draw.rect(windowSurfaceObj, blackColor, (y*u, x*u, u, u))

def calcularCanny(image, sigma=0.33):

    v = np.median(image)
    
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

def disponibilidadPlaza(img):

    estado = False
    edged = calcularCanny(img)
    blancos = cv2.countNonZero(edged)
    if (blancos >= 450):
        estado = True
    
    return estado

def transformarPerspectiva(SupIz,SupDer,InfIz,InfDer):
    
    pts1 = np.float32([SupIz, SupDer, InfIz, InfDer])
    pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (640, 480))
    
    return result

"""
---------------------------------------------------------
"""
##GPIO.setmode(GPIO.BCM)
##GPIO.setwarnings(False)
##
##GPIO.setup(17,GPIO.OUT)
##GPIO.setup(27,GPIO.OUT)
##GPIO.setup(20,GPIO.OUT)
##GPIO.setup(21,GPIO.OUT)
##GPIO.setup(12,GPIO.OUT)
##GPIO.setup(16,GPIO.OUT)
##GPIO.setup(7,GPIO.OUT)
##GPIO.setup(8,GPIO.OUT)

# Define GPIO to LCD mapping
##LCD_RS = 26
##LCD_E  = 19
##LCD_D4 = 13
##LCD_D5 = 6
##LCD_D6 = 5
##LCD_D7 = 11

#GPIO.setmode(GPIO.BCM)       # Use BCM GPIO numbers
#GPIO.setup(LCD_E, GPIO.OUT)  # E
#GPIO.setup(LCD_RS, GPIO.OUT) # RS
#GPIO.setup(LCD_D4, GPIO.OUT) # DB4
#GPIO.setup(LCD_D5, GPIO.OUT) # DB5
#GPIO.setup(LCD_D6, GPIO.OUT) # DB6
#GPIO.setup(LCD_D7, GPIO.OUT) # DB7

##lcd = CharLCD(numbering_mode=GPIO.BCM,cols=16, rows=2, pin_rs=LCD_RS, pin_e=LCD_E, pins_data=[LCD_D4, LCD_D5, LCD_D6, LCD_D7])
##
##lcd.clear()
##lcd.cursor_pos = (0, 0)
##lcd.write_string('Slot1:O Slot2:O')
##lcd.cursor_pos = (1, 0)
##lcd.write_string('Slot3:O Slot4:O')
#time.sleep(2)

##try:
##    firebase = firebase.FirebaseApplication('https://parqueadero-73a08.firebaseio.com/')
##except:
##    pass

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('/home/pi/Desktop/Avanzado/ouyttputt.mp4')

maze = [[1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1]]

#Caracteristicas de color
#whiteColor = pygame.Color(255,255,255)
#blackColor = pygame.Color(0,0,0)
#greenColor = pygame.Color(0,255,0)

#Alto y ancho de la imagen, esta al revez
#mazeWidth = 6
#mazeHeight = 5
#
#u= 30
#
#windowWidth = mazeWidth * u
#windowHeight = mazeHeight * u
#windowSurfaceObj = pygame.display.set_mode((windowWidth,windowHeight))

xanterior = 0
path_anterior=0

x =  { "Slot1": True,
       "Slot2": True,
       "Slot3": True,
       "Slot4": True,
       "path": 0 }

##iniciarMapa()

while True:
    
    _, frame = cap.read()
    frame = cv2.blur(frame,(3,3))
    frame = cv2.resize(frame,(np.int(frame.shape[1]/2),np.int(frame.shape[0]/2)))

    #Ciudad
    cv2.circle(frame, (50, 2), 2, (255, 255, 0), -1)
    cv2.circle(frame, (295, 2), 2, (255, 255, 0), -1)
    cv2.circle(frame, (-35, 230), 2, (255, 255, 0), -1)
    cv2.circle(frame, (385, 230), 2, (255, 255, 0), -1)

    Ciudad = transformarPerspectiva([50, 2],[295, 2],[-35, 230],[385, 230])
    Ciudad2 = Ciudad.copy()
    CiudadGray = cv2.cvtColor(Ciudad, cv2.COLOR_BGR2GRAY)    

    ##Rectangulos donde estan los nodos
    cv2.rectangle(Ciudad2, (250, 305), (320, 355), (255,255,0), 2)
    cv2.rectangle(Ciudad2, (320, 305), (385, 355), (255,255,0), 2)
    cv2.rectangle(Ciudad2, (105, 355), (180, 410), (255,255,0), 2)
    cv2.rectangle(Ciudad2, (180, 355), (250, 410), (255,255,0), 2)
    cv2.rectangle(Ciudad2, (250, 355), (320, 410), (255,255,0), 2)
    cv2.rectangle(Ciudad2, (320, 355), (385, 410), (255,255,0), 2)

    ##NODOS
    nodo1=Ciudad[305:355,250:320]
    nodo2=Ciudad[305:355,320:385]    
    nodo3=Ciudad[355:410,105:180]
    nodo4=Ciudad[355:410,180:250]
    nodo5=Ciudad[355:410,250:320]
    nodo6=Ciudad[355:410,320:385]
##    nodoF1=Ciudad[410:475,105:180]
##    nodoF2=Ciudad[410:475,180:250]
##    nodoF3=Ciudad[410:475,250:320]
##    nodoF4=Ciudad[410:475,320:385]

    #Region de interes, los slots
    slot1=CiudadGray[410:475,105:180]
    slot2=CiudadGray[410:475,180:250]
    slot3=CiudadGray[410:475,250:320]    
    slot4=CiudadGray[410:475,320:385]

    #True o False el estado de los slots del estacionamiento
    estado1 = disponibilidadPlaza(slot1)
    estado2 = disponibilidadPlaza(slot2)
    estado3 = disponibilidadPlaza(slot3)
    estado4 = disponibilidadPlaza(slot4)

    ##Enviamos la region de interes del nodo, su respectiva posición
    ##en la matriz y los estados de los slots
    path1,rutaS1 = dibujarRuta(nodo1,(1,3),estado1,estado2,estado3,estado4)
    path2,rutaS2 = dibujarRuta(nodo2,(1,4),estado1,estado2,estado3,estado4)
    path3,rutaS3 = dibujarRuta(nodo3,(2,1),estado1,estado2,estado3,estado4)
    path4,rutaS4 = dibujarRuta(nodo4,(2,2),estado1,estado2,estado3,estado4)
    path5,rutaS5 = dibujarRuta(nodo5,(2,3),estado1,estado2,estado3,estado4)
    path6,rutaS6 = dibujarRuta(nodo6,(2,4),estado1,estado2,estado3,estado4)

    lcd.clear()
    #Dibujamos verde si se encuentra disponible, Rojo si se encuentra ocupado
    if(estado1==False):
        cv2.rectangle(Ciudad2, (105, 410), (180, 475), (0,255,0), 2)
##        lcd.cursor_pos = (0, 0)
##        lcd.write_string('Slot1:O')
##        GPIO.output(12,GPIO.HIGH)
##        GPIO.output(16,GPIO.LOW)
##        
    else:
        cv2.rectangle(Ciudad2, (105, 410), (180, 475), (0,0,255), 2)
##        lcd.cursor_pos = (0, 0)
##        lcd.write_string('Slot1:X')
##        GPIO.output(16,GPIO.HIGH)
##        GPIO.output(12,GPIO.LOW)

    if(estado2==False):
        cv2.rectangle(Ciudad2, (180, 410), (250, 475), (0,255,0), 2)
##        lcd.cursor_pos = (0, 8)
##        lcd.write_string('Slot2:O')
##        GPIO.output(20,GPIO.HIGH)
##        GPIO.output(21,GPIO.LOW)
        
    else:
        cv2.rectangle(Ciudad2, (180, 410), (250, 475), (0,0,255), 2)
##        lcd.cursor_pos = (0, 8)
##        lcd.write_string('Slot2:X')
##        GPIO.output(21,GPIO.HIGH)
##        GPIO.output(20,GPIO.LOW)

    if(estado3==False):
        cv2.rectangle(Ciudad2, (250, 410), (320, 475), (0,255,0), 2)
##        lcd.cursor_pos = (1, 0)
##        lcd.write_string('Slot3:O')
##        GPIO.output(17,GPIO.HIGH)
##        GPIO.output(27,GPIO.LOW)
        
    else:
        cv2.rectangle(Ciudad2, (250, 410), (320, 475), (0,0,255), 2)
##        lcd.cursor_pos = (1, 0)
##        lcd.write_string('Slot3:X')
##        GPIO.output(27,GPIO.HIGH)
##        GPIO.output(17,GPIO.LOW)

    if(estado4==False):
        cv2.rectangle(Ciudad2, (320, 410), (385, 475), (0,255,0), 2)
##        lcd.cursor_pos = (1, 8)
##        lcd.write_string('Slot4:O')
##        GPIO.output(7,GPIO.HIGH)
##        GPIO.output(8,GPIO.LOW)
        
    else:
        cv2.rectangle(Ciudad2, (320, 410), (385, 475), (0,0,255), 2)
##        lcd.cursor_pos = (1, 8)
##        lcd.write_string('Slot4:X')
##        GPIO.output(8,GPIO.HIGH)
##        GPIO.output(7,GPIO.LOW)

#    time.sleep(1)
    #Genero un archivo JSON con el estado de los slots y la ruta
    #que debe tomar el carrito dependiendo el nodo en el que se encuentre
#    print(rutaS1)

    if(path1 != 0):        
        x =  { "Slot1": estado1,
               "Slot2": estado2,
               "Slot3": estado3,
               "Slot4": estado4,
               "path": rutaS1 }
        path_anterior=rutaS1
        
    elif(path2 != 0):
        x =  { "Slot1": estado1,
               "Slot2": estado2,
               "Slot3": estado3,
               "Slot4": estado4,
               "path": rutaS2 }
        path_anterior=rutaS2
        
    elif(path3 != 0):
        x =  { "Slot1": estado1,
               "Slot2": estado2,
               "Slot3": estado3,
               "Slot4": estado4,
               "path": rutaS3 }
        path_anterior=rutaS3
        
    elif(path4 != 0):
        x =  { "Slot1": estado1,
               "Slot2": estado2,
               "Slot3": estado3,
               "Slot4": estado4,
               "path": rutaS4 }
        path_anterior=rutaS4
        
    elif(path5 != 0):
        x =  { "Slot1": estado1,
               "Slot2": estado2,
               "Slot3": estado3,
               "Slot4": estado4,
               "path": rutaS5 }
        path_anterior=rutaS5
        
    elif(path6 != 0):
        x =  { "Slot1": estado1,
               "Slot2": estado2,
               "Slot3": estado3,
               "Slot4": estado4,
               "path": rutaS6 }
        path_anterior=rutaS6
        
    else:
        x =  { "Slot1": estado1,
               "Slot2": estado2,
               "Slot3": estado3,
               "Slot4": estado4,
               "path": 0 }
        
#        iniciarMapa()

##    try:
##        if(x != xanterior):
##                result = firebase.put('/user','Parking',x)
##    except:
##        pass
            
    gc.collect()
    xanterior = x

##    with open('Info.json', 'w') as json_file:  
##        json.dump(x, json_file)
##    print(json.dumps(x))
    
#    cv2.imshow("Frame", frame)
#    cv2.imshow("Ciudad", Ciudad2)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
##    cv2.waitKey(200)

cap.release()
cv2.destroyAllWindows()
