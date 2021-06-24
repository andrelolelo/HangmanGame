import random
import collections
import pyaudio
from gtts import gTTS
import os
import speech_recognition as sr
import nltk
from nltk import grammar, parse
from nltk import load_parser
import face_recognition
import cv2
import numpy as np
import glob
import os
import logging
import random
from pyDatalog import pyDatalog
from colorama import init, Fore, Back, Style
from playsound import playsound

# Ruta donde se obtienen las imágenes
images_path = './images'   

# Dispositivo webcam
camera_device_id = 0

MAX_DISTANCE = 0.6 

# Reconocimiento facial
def get_face_embeddings_from_image(image, convert_to_rgb=False):
    
    # Conversión de imagen con formarto BGR a RGB 
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # Ejecuta el modelo de detección de rostros para encontrar ubicaciones de rostros
    face_locations = face_recognition.face_locations(image, model="hog")

    # Ejecute el modelo de incrustación para obtener incrustaciones faciales para las ubicaciones proporcionadas
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings


# Carga las imágenes de referencia y crea una base de datos de sus codificaciones faciales
def setup_database():
 
    database = {}

    for filename in glob.glob(os.path.join(images_path, '*.jpg')):
        # Cargar imagen
        image_rgb = face_recognition.load_image_file(filename)

        # Usa el nombre del archivo como clave de identidad
        identity = os.path.splitext(os.path.basename(filename))[0]
        name = (identity)

        # Obtiene la codificación de la cara y la vincula a la identidad
        locations, encodings = get_face_embeddings_from_image(image_rgb)
        database[name] = encodings[0]
        
    return database


# Función que plasma un rectángulo alrededor del rostro
def paint_detected_face_on_image(frame, location, name=None):
    
    # Descomprime las coordenadas de la tupla de ubicación
    top, right, bottom, left = location

    if name is None:
        name = 'No tenemos información tuya en la base de datos'
        # Color rojo para rostros desconocidos
        color = (0, 0, 255)  # rojo para cara no reconocida
    else:
        # Color verde para rostros reconocidos
        color = (124, 252, 0)  

    # Cuadro alrededor del rostro
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Etiqueta con un nombre debajo del rostro
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)


# Función que ejecuta el reconocimiento facial a través de la webcam
def run_face_recognition(database):
 
    # Se abre controlador para la webcam
    video_capture = cv2.VideoCapture(camera_device_id)

    # the face_recognitino library uses keys and values of your database separately
    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())
    
    mensajeReconocimientoFacial = "A continuación, el juego llevará a cabo el reconocimiento facial del usuario."
    vozAsistente (mensajeReconocimientoFacial)
    print("\n    Ahora, el juego iniciará el reconocimiento facial... \n")
    
    while video_capture.isOpened():
        # Grab a single frame of video (and check if it went ok)
        ok, frame = video_capture.read()
        if not ok:
            logging.error("Could not read frame from camera. Stopping video capture.")
            break

        # run detection and embedding models
        face_locations, face_encodings = get_face_embeddings_from_image(frame, convert_to_rgb=True)

        # Loop through each face in this frame of video and see if there's a match
        for location, face_encoding in zip(face_locations, face_encodings):

            # get the distances from this encoding to those of all reference images
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # select the closest match (smallest distance) if it's below the threshold value
            if np.any(distances <= MAX_DISTANCE):
                best_match_idx = np.argmin(distances)
                name = known_face_names[best_match_idx]
            else:
                name = None
                mensajeReconocimientoFacialError = "No logro reconocer quién eres tú. Registra tu foto. "
                print("Usuario desconocido. Registra tu foto. ")
                vozAsistente(mensajeReconocimientoFacialError)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit()
                video_capture.release()
                cv2.destroyAllWindows()

            # put recognition info on the image
            paint_detected_face_on_image(frame, location, name)
            print("Hola, tú eres: "+str(name)+", pulsa la tecla 'a' para continuar.")
            if cv2.waitKey(1) & 0xFF == ord('a'):
                video_capture.release()
                cv2.destroyAllWindows()
                callAgent(name)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

# Animaciones de figura del juego del ahorcado

figuraAhorcado = ['''

   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄
   ║           \█
                █
                █
                █
                █
                █
   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒''', '''
   
   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄
   ║           \█
   ☻            █
                █
                █
                █
                █
   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒''', '''
   
   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄
   ║           \█
   ☻            █
   |            █
                █
                █
                █
   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒''', '''
   
   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄
   ║           \█
   ☻            █
  /|            █
                █
                █
                █
   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒''', '''
   
   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄
   ║           \█
   ☻            █
  /|\           █
                █
                █
                █
   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒''', '''

   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄
   ║           \█
   ☻            █
  /|\           █
  /             █
                █
                █
   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒''', '''
   
   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄
   ║           \█
   ☻            █
  /|\           █
  / \           █
                █
                █
   ▒▒▒▒▒▒▒▒▒▒▒▒▒▒''']


# Listado de palabras: dificultad baja, media y alta
palabraDificultadBaja = 'perro fresa'.split()
palabraDificultadMedia = 'escaparate'.split()
palabraDificultadAlta = 'anticonstitucional'.split()

# Función que regresa una cadena al azar de la lista de palabras
def obtenerPalabra(listaDePalabras):
    indiceDePalabras = random.randint(0, len(listaDePalabras) - 1)
    return listaDePalabras[indiceDePalabras]
 
# Función que se encarga de imprimir en pantalla el tablero del juego del ahorcado
def mostrarTablero(figuraAhorcado, letrasIncorrectas, letrasCorrectas, palabraSecreta):
    print(figuraAhorcado[len(letrasIncorrectas)])
    print()
 
    print('Letras incorrectas:', end=' ')
    for letra in letrasIncorrectas:
        print(letra, end=' ')
    print()
 
    espaciosVacíos = '_' * len(palabraSecreta)
 
    # Completar los espacios vacíos con las letras adivinadas
    for i in range(len(palabraSecreta)): 
        if palabraSecreta[i] in letrasCorrectas:
            espaciosVacíos = espaciosVacíos[:i] + palabraSecreta[i] + espaciosVacíos[i+1:]
 
    # Mostrar la palabra secreta con espacios entre cada letra
    for letra in espaciosVacíos: 
        print(letra, end=' ')
    print()
 
def GRA(voz):
    try:
        cp = parse.load_parser('sem2.fcfg', trace=1)
        tokens = voz.split()
        rd_parser = cp.parse(tokens)
        regresa= ""
        
        for tree in rd_parser:
            if tree:
                regresa= tokens[-1]
                return regresa
    except ValueError: 
        print("No entra en la gramatica")
            
def obtenerIntento(letrasProbadas):
    # Devuelve la letra ingresada por el jugador. Verifica que el jugador ha ingresado sólo una letra, y no otra cosa.
    r = sr.Recognizer()
    mic = sr.Microphone()
    verdad = True
    while verdad == True:
        print('Piensa en una letra y dilo en voz alta...')
        mensajeLetra = "Piensa en una letra y dilo en voz alta..."
        vozAsistente(mensajeLetra)
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source) 
            voz = r.recognize_google(audio,language = 'es-ES')
            #print("I thinks you said '" + r.recognize_google(audio) + "'")
            intento = GRA(voz)
            if intento != None:
                return (intento.lower())
                verdad = False
            else:
                errorIntento="Lo que has dicho, no entra en la gramática. "
                print(errorIntento)
                vozAsistente(errorIntento)
                #Verdad = True
            
 
def jugarDeNuevo():
    # Esta función devuelve True si el jugador quiere volver a jugar, en caso contrario devuelve False.
    print('¿Quieres jugar de nuevo? (s o n)')
    return input().lower().startswith('s')
 

# Agente

def callAgent(valor):
    name = valor
    # Objeto de tipo Trivial
    e = TrivialHangmanEnvironment()
    
    #Objeto agente "ReflexHangmanAgent"
    a = ReflexHangmanAgent(name)     
    
    # Encargado de observar lo que hace el agente
    TraceAgent(a)    
        
    # Se agrega al agente en el ambiente        
    e.add_thing(a)       

    # Se asigna que en donde está el ambiente se le asigna al agente, haciendo una relación      
    a.setEnvironment(e)             
        
    k=1
    while (k<2):
        # El agente detecta las cosas del ambiente
        percept = a.sense();          

        # Determina el comportamiento del agente    
        action = a.selectAction(percept);     
        
        # Indica al ambiente que se ejecute una acción
        e.execute_action(a,action);             
        k += 1

# Clase Thing

#Simboliza cualquier objeto que pueda aparecer en el ambiente
class Thing:
    """This represents any physical object that can appear in an Environment.
    You subclass Thing to get the things you want. Each thing can have a
    .__name__  slot (used for output only)."""

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def is_alive(self):
        """Things that are 'alive' should return true."""
        return hasattr(self, 'alive') and self.alive

    # Despliega el estado interno del agente
    def show_state(self):
        print("I don't know how to show_state.")

    def display(self, canvas, x, y, width, height):
        """Display an image of this Thing on the canvas."""
        # Do we need this?
        pass

# Clase Agente
class Agent(Thing):

    def __init__(self, program=None):
        self.alive = True
        self.bump = False
        self.holding = []
        self.performance = 0
        
        if program is None or not isinstance(program, collections.Callable):
            print("Can't find a valid program for {}, falling back to default.".format(
                self.__class__.__name__))

            def program(percept):
                return eval(input('Percept={}; action? '.format(percept)))

        self.program = program

    def setEnvironment(self,env):
        self.environment=env
    
    # Verifica la percepción
    def sense(self):
        return(self.environment.percept(self))
        
    # Regresa el programa con la percepción
    def selectAction(self, percept):
        return self.program(percept)


def TraceAgent(agent):
    """Wrap the agent's program to print its input and output. This will let
    you see what the agent is doing in the environment."""
    old_program = agent.program

    def new_program(percept):
        action = old_program(percept)
        print('{} perceives {} and does {}'.format(agent, percept, action))
        return action
    agent.program = new_program
    return agent

# Función que comprende los estereotipos
def stereotypes():
    
    pyDatalog.create_terms('Edad',
                       'Nombre',
                       'Experiencia',
                       'dificultad_baja',
                       'dificultad_media',
                       'dificultad_alta',
                       'baja',
                       'media',
                       'alta',
                       'tiene_experiencia_leve',
                       'tiene_experiencia_intermedia',
                       'tiene_experiencia_alta')
    
    pyDatalog.load('dificultad_baja(Nombre)<= baja(Nombre) & tiene_experiencia_leve(Nombre)')
    pyDatalog.load('dificultad_media(Nombre)<= media(Nombre) & tiene_experiencia_intermedia(Nombre)')
    pyDatalog.load('dificultad_alta(Nombre)<= alta(Nombre) & tiene_experiencia_alta(Nombre)')
    
    
    # Bloque de diálogos del asistente de voz
    
    instruccionesA = "Ahora, es necesario que contestes las siguientes sentencias. Únicamente son para determinar la dificultad el juego."
    instruccionesIniciales = instruccionesA
    
    instruccionesB = "Para interactuar con el juego, debes preparar tu voz y hablar en voz alta."
    instruccionesC = "Ahora te diré las frases que puedes decir."
    instruccionesD = "Escribe la letra equis."
    instruccionesE = "Ingresa la letra equis."
    instruccionesF = "Obviamente, la equis es un ejemplo, por lo que la equis puedes sustituirla por cualquier letra."
    instruccionesIniciales2 = "Instrucciones iniciales."+instruccionesB+instruccionesC+instruccionesD+instruccionesE+instruccionesF
    
    instruc="""
    ╔═════════════════════════════════════════════════════╗                      
    ║                                                     ║ 
    ║                    Instrucciones                    ║
    ║                                                     ║
    ║   El jugador puede decir las siguientes frases:     ║
    ║     ■ Escribe la letra 'x'                          ║
    ║     ■ Ingresa la letra 'x'                          ║                         
    ║                                                     ║
    ╚═════════════════════════════════════════════════════╝
    
    ╔═════════════════════════════════════════════════════╗                      
    ║                                                     ║ 
    ║        ▒▒▒▒▒▒▒                                      ║
    ║        |~   ~|       ＿＿＿＿＿＿＿＿＿              ║
    ║        |◉   ◉|     /                   \            ║                 
    ║        |  ʖ  |    <  Escribe la letra a |           ║       
    ║         \ O /      \ ＿＿＿＿＿＿＿＿＿ /            ║                      
    ║          | |                                        ║                                                                   
    ╚═════════════════════════════════════════════════════╝
    """
    print(instruc)
    vozAsistente(instruccionesIniciales)
    vozAsistente(instruccionesIniciales2)

def ReflexHangmanAgent(name):
    
    def program(percept):
        location, status = percept
        if status == 'letra': 
            mensajeComienzo = "¡Vamos a jugar al ahorcado!"
            print("\n    Inicio exitoso... \n")
            vozAsistente(mensajeComienzo)
            letrasIncorrectas = ''
            letrasCorrectas = ''
            
            stereotypes()
            validate = True
            
            while validate == True:
                print("\n    Información del usuario: \n")
                
                
                Nombre = name 
                di = "\n    Nombre del jugador: "+ Nombre
                print(di)
                mensajeNombreUsuario = "Yo sé quién eres, tu nombre es " + Nombre
                vozAsistente(mensajeNombreUsuario)
                
                
                mensajeEdad = "Es momento de que me digas cuántos años tienes."
                vozAsistente(mensajeEdad)     
                Edad = int(input( '\n    Por favor, ingresa tu edad: '))
                
                mensajeExperiencia = "Gracias por la información. Ahora me gustaría conocer cuál es tu experiencia en videojuegos. Tu experiencia puede ser leve, intermedia o alta."
                vozAsistente(mensajeExperiencia)
                tablaExperiencia="""
                ╔═════════════════════════════════════════════════════╗                      
                ║                                                     ║ 
                ║                     ( ͡° ͜ʖ ͡°)                        ║
                ║                                                     ║
                ║   ¿Cuál es tu experiencia en videojuegos:           ║
                ║     ■ Leve                                          ║
                ║     ■ Intermedia                                    ║                         
                ║     ■ Alta                                          ║
                ╚═════════════════════════════════════════════════════╝
                
                """
                print(tablaExperiencia)
                Experiencia = input( '\n    Por favor, ingresa tu experiencia en videojuegos:  ' )
                
                if Edad>=7 and Edad<=12:
                    pyDatalog.assert_fact('baja', Nombre)
                    if Experiencia == 'Leve':
                        pyDatalog.assert_fact('tiene_experiencia_leve', Nombre)
                        baja = pyDatalog.ask('dificultad_baja(Nombre)').answers[0][0]
                        vozAsistente("Te he asignado la dificultad baja")
                        print("\n    Dificultad leve asignada " + baja)
                        palabraSecreta = obtenerPalabra(palabraDificultadBaja)
                        juegoTerminado = False
                        validate = False
                    else: 
                        print("\n    No eres apto para jugar este videojuego. \n")
                        vozAsistente("De acuerdo a tus respuestas, creo que no estás preparado para jugar este juego. Lo siento, adiós.")
                        
                elif Edad >=13 and Edad <= 18:
                    pyDatalog.assert_fact('media', Nombre)
                    if Experiencia == 'Intermedia':
                        pyDatalog.assert_fact('tiene_experiencia_intermedia', Nombre)
                        media = pyDatalog.ask('dificultad_media(Nombre)').answers[0][0]
                        vozAsistente("Te he asignado la dificultad media ")
                        print("\n    Dificultad media asignada " + media)
                        palabraSecreta = obtenerPalabra(palabraDificultadMedia)
                        juegoTerminado = False
                        validate = False
                    else:
                        print("\n    No eres apto para jugar este videojuego. \n")
                        vozAsistente("De acuerdo a tus respuestas, creo que no estás preparado para jugar este juego. Lo siento, adiós.")
                        
                elif Edad >=19:
                    pyDatalog.assert_fact('alta', Nombre)
                    if Experiencia == 'Alta':
                        pyDatalog.assert_fact('tiene_experiencia_alta', Nombre)
                        alta = pyDatalog.ask('dificultad_alta(Nombre)').answers[0][0]
                        vozAsistente("Te he asignado la dificultad alta ")
                        print("\n    Dificultad alta asignada " + alta)
                        palabraSecreta = obtenerPalabra(palabraDificultadAlta)
                        juegoTerminado = False
                        validate = False
                    else:
                        print("\n    No eres apto para jugar este videojuego. \n")
                        vozAsistente("De acuerdo a tus respuestas, creo que no estás preparado para jugar este juego. Lo siento, adiós.")
                
                else:
                    print("\n    No cumpliste con los criterios necesarios para jugar este videojuego... \n")
                    vozAsistente("Quizás sea tu edad o tu experiencia en videojuegos, pero por tales razones no puedes jugar a este videojuego.")
                    
            while True:
                mostrarTablero(figuraAhorcado, letrasIncorrectas, letrasCorrectas, palabraSecreta)
             
                # Permite al jugador escribir una letra.
                unaletra = obtenerIntento(letrasIncorrectas + letrasCorrectas)
                print(unaletra)
             
                if unaletra in palabraSecreta:
                    letrasCorrectas = letrasCorrectas + unaletra
                    correc = "¡Muy bien! la letra, "+ unaletra + ", sí está en la palabra secreta. Sigue así."
                    vozAsistente(correc)
             
                    # Verifica si el jugador ha ganado.
                    encontradoTodasLasLetras = True
                    for i in range(len(palabraSecreta)):
                        if palabraSecreta[i] not in letrasCorrectas:
                            encontradoTodasLasLetras = False
                            break
                        
                    if encontradoTodasLasLetras:
                        print('¡Sí! ¡La palabra secreta es "' + palabraSecreta + '"! ¡Has ganado!')
                        mensajeFinal= "Sí, La palabra secreta es, "+ palabraSecreta + ".  Has ganado Jugador, "+ name
                        vozAsistente(mensajeFinal)
                        juegoTerminado = True
                else:
                    letrasIncorrectas = letrasIncorrectas + unaletra
                    error = "¡Ups! la letra, "+ unaletra + ", no está en la palabra secreta. Intenta de nuevo."
                    vozAsistente(error)
             
                    # Comprobar si el jugador ha agotado sus intentos y ha perdido.
                    if len(letrasIncorrectas) == len(figuraAhorcado) - 1:
                        mostrarTablero(figuraAhorcado, letrasIncorrectas, letrasCorrectas, palabraSecreta)
                        print('¡Te has quedado sin intentos!\nDespués de ' + str(len(letrasIncorrectas)) + ' intentos fallidos y ' + str(len(letrasCorrectas)) + ' aciertos, la palabra era "' + palabraSecreta + '"')
                        vozAsistente("¡Demonios! Te has quedado sin intentos, la palabra secreta era"+ palabraSecreta + "Creí que era un buen jugador. Suerte para la próxima.")
                        juegoTerminado = True
             
                # Preguntar al jugador si quiere volver a jugar (pero sólo si el juego ha terminado).
                if juegoTerminado:
                    if jugarDeNuevo():
                        letrasIncorrectas = ''
                        letrasCorrectas = ''
                        juegoTerminado = False
                        palabraSecreta = obtenerPalabra(palabraDificultadMedia)
                    else:
                        break
                    return 'escucha'
            return 'escucho'
    return Agent(program)

# Environment

# Clase que representa un ambiente
# Subclase del paquete EasyAI

class Environment:

    def __init__(self):
        self.things = []
        self.agents = []

    # Lista de clases que pueden ir en el ambiente
    def thing_classes(self):
        return []  

    # Regresa la percepción que el agente logra ver
    def percept(self, agent):
        raise NotImplementedError

    # Lleva a cabo la acción
    def execute_action(self, agent, action):
        """Change the world to reflect this action. (Implement this.)"""
        raise NotImplementedError

    # Asigna la locación por defecto de la cosa
    def default_location(self, thing):
        """Default location to place a new thing with unspecified location."""
        return None

    # Cuando no se encuentre un agente vivo, se termina todo.
    def is_done(self):
        """By default, we're done when we can't find a live agent.
        is_alive verifica si un agente esta vivo"""
        return not any(agent.is_alive() for agent in self.agents)

    # Ejecuta el ambiente por un paso de tiempo
    def step(self):
        
        # Si el agente está vivo, entocnes se le asigna la percepción
        if not self.is_done():
            actions = []
            for agent in self.agents:
                if agent.alive:
                    actions.append(agent.program(self.percept(agent)))
                else:
                    actions.append("")
            for (agent, action) in zip(self.agents, actions):
                self.execute_action(agent, action)

    def run(self, steps=1000):
        """Run the Environment for given number of time steps.
        steps = 1000 indica un valor por defecto si no se le pasa un parametro"""
        for step in range(steps):
            if self.is_done():
                return
            self.step()

    def list_things_at(self, location, tclass=Thing):
        """Return all things exactly at a given location."""
        return [thing for thing in self.things
                if thing.location == location and isinstance(thing, tclass)]

    def some_things_at(self, location, tclass=Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        return self.list_things_at(location, tclass) != []

    # Método que se encarga de colocar Thing en algún sitio
    def add_thing(self, thing, location=None):
        """Add a thing to the environment, setting its location. For
        convenience, if thing is an agent program we make a new agent
        for it. (Shouldn't need to override this.)
        Verifica si lo que estamos pasando es una cosa, para verificar si no es un agente"""
        if not isinstance(thing, Thing):
            thing = Agent(thing)
        if thing in self.things:
            print("Can't add the same thing twice")
        else:
            thing.location = location if location is not None else self.default_location(thing)
            self.things.append(thing)
            if isinstance(thing, Agent):
                thing.performance = 0
                self.agents.append(thing)

    # Elimina la cosa del ambiente
    def delete_thing(self, thing):
        try:
            self.things.remove(thing)
        except ValueError as e:
            print(e)
            print("  in Environment delete_thing")
            print("  Thing to be removed: {} at {}".format(thing, thing.location))
            print("  from list: {}".format([(thing, thing.location) for thing in self.things]))
        if thing in self.agents:
            self.agents.remove(thing)


# TrivialHangmanEnvironment

# Aquí se encuentra la implementación del ambiente, tomando como parámetro Environment
# Describe el mundo en el que se establece el juego
# Antes llamado GameController

class TrivialHangmanEnvironment(Environment):

    
    def __init__(self):
        super().__init__()
        # Definición de estado A
        self.status = {loc_A: random.choice(['letra'])} 

    def thing_classes(self):
        return [ReflexHangmanAgent]
        
    def percept(self, agent):
        """Returns the agent's location, and the location status (Dirty/Clean)."""
        return (agent.location, self.status[agent.location])

    # Ejecutar la acción, es decir, define los efectos de ejecutar una acción
    # Debido a que se adapta el contexto del agente limpiador se cambia de "make_move" a "execute_action"
    def execute_action(self, agent, action):
        """Change agent's location and/or location's status; track performance."""
        if action == 'letra':
            print("Accion decir Letra")
            agent.location = loc_A 

    # Se establece el estado loc_A como por defecto
    def default_location(self, thing):
        """Agents start in either location at random."""
        return random.choice([loc_A]) 

# Definición de las locaciones donde estará el ambiente
loc_A = 'A'  


def vozAsistente(text):
    tts = gTTS(text, lang='es-ES') 
    tts.save("ahorcado.mp3")
    playsound("ahorcado.mp3", True)
    os.remove("ahorcado.mp3")   
    #os.system("mpg123 ahorcado.mp3")   


def inicio():
    textoInicioJuego="""
    ╔═════════════════════════════════════════════════════╗                      
    ║             _                             _         ║
    ║       /\   | |                           | |        ║
    ║      /  \  | |__   ___  _ __ ___ __ _  __| | ___    ║
    ║     / /\ \ | '_ \ / _ \| '__/ __/ _` |/ _` |/ _ \   ║
    ║    / ____ \| | | | (_) | | | (_| (_| | (_| | (_) |  ║
    ║   /_/    \_\_| |_|\___/|_|  \___\__,_|\__,_|\___/   ║                          
    ║                                                     ║
    ╚═════════════════════════════════════════════════════╝
    """
    
    print (textoInicioJuego)
    mensajeinicio = "Bienvenido al juego del miedo, digo, del ahorcado."
    vozAsistente(mensajeinicio)
    
    # Accede a la base de datos de imágenes
    database = setup_database() 
    
    # Ejecuta el reconocimiento de rostro
    x = run_face_recognition(database)
    return (x)

if __name__ == "__main__":
    inicio()


    