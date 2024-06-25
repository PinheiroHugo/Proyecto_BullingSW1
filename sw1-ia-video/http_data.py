import requests


def crear_evidencia_video(video_path,id_evento):
    # URL de ejemplo
    # url = 'http://104.198.243.205/api/evidencia'
    #url = 'http://127.0.0.1:8001/api/evidencia'
    url = 'http://192.241.150.219/api/evidencia'

    # Encabezados (headers)
    headers = {
        # 'Accept': 'application/json',
        'Authorization': 'Bearer your_access_token'  # Opcional: agregar un encabezado de autorización
    }

    # Datos para el cuerpo (body) de la solicitud en formato JSON
    data = {
        'evento_id': id_evento,
        'tipo': 'Video',
    }

    # Abre el archivo de video en modo binario
    with open(video_path, 'rb') as video_file:
        # Crear un diccionario de archivos para enviar el archivo
        files = {'file': open(video_path,'rb')}

        # Realizar una solicitud POST con encabezados y cuerpo
        response = requests.post(url, data=data, json=data, headers=headers, files=files)


        # Verificar si la solicitud fue exitosa (código de estado 201 para una creación exitosa)
        if response.status_code >= 200 and response.status_code <300:
            # Obtener la respuesta en formato JSON
            created_post = response.json()
            print(created_post)
            return created_post
        else:
            print(f"La solicitud POST falló con el código de estado {response.status_code}")
            print(response.json())
            return False


def crear_evento(descripcion):
    # URL de ejemplo
    #url = 'http://104.198.243.205/api/evento'
    #url = 'http://127.0.0.1:8001/api/evento'
    url = 'http://192.241.150.219/api/evento'

    # Encabezados (headers)
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer your_access_token'  # Opcional: agregar un encabezado de autorización
    }

    # Datos para el cuerpo (body) de la solicitud en formato JSON
    data = {
        'fecha': '2023/11/05',
        'descripcion': descripcion,
        'camara_id': 1,
        'es_queja': 0
    }

    # Realizar una solicitud POST con encabezados y cuerpo
    response = requests.post(url, json=data, headers=headers)

    # Verificar si la solicitud fue exitosa (código de estado 201 para una creación exitosa)
    if response.status_code >= 200 and response.status_code <300:
        # Obtener la respuesta en formato JSON
        created_post = response.json()
        print(created_post)
        return created_post
    else:
        print(f"La solicitud POST falló con el código de estado {response.status_code}")
        print(response.json())
        return False

