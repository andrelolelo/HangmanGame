from pyDatalog import pyDatalog

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

validate = True
while validate == True:
    Edad = int(input('Por favor, ingresa tu edad: '))
    Nombre = input('Por favor, ingresa tu nombre: ')
    Experiencia = input('¿Cuál es tu experiencia en videojuegos?: ')
    
    if Edad>=7 and Edad<=12 and Experiencia == 'Leve':
        pyDatalog.assert_fact('baja', Nombre)
        pyDatalog.assert_fact('tiene_experiencia_leve', Nombre)
        baja = pyDatalog.ask('dificultad_baja(Nombre)').answers[0][0]
        print("Dificultad baja asignada para el usuario " + baja)
        validate = False
        
    elif Edad >=13 and Edad <= 18 and Experiencia == 'Intermedia':
        pyDatalog.assert_fact('media', Nombre)
        pyDatalog.assert_fact('tiene_experiencia_intermedia', Nombre)
        media = pyDatalog.ask('dificultad_media(Nombre)').answers[0][0]
        print("Dificultad media asignada para el usuario " + media)
        validate = False
        
    elif Edad >= 19 and Experiencia == 'Alta':
        pyDatalog.assert_fact('alta', Nombre)
        pyDatalog.assert_fact('tiene_experiencia_alta', Nombre)
        alta = pyDatalog.ask('dificultad_alta(Nombre)').answers[0][0]
        print("Dificultad alta asignada para el usuario " + alta)
        validate = False
        
    else:
        print("No eres apto para jugar este videojuego.")
