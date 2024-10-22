import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.video import Video
from kivy.animation import Animation
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
import os

import kivy
from kivy.app import App
from kivy.uix.video import Video
from kivy.uix.relativelayout import RelativeLayout

class MiApp(App):
    def build(self):
        layout = RelativeLayout()
        video = Video(source='Python/PRUEBA/ANIMACION APP.mp4', play=True, allow_stretch=True)
        layout.add_widget(video)
        return layout

if __name__ == '__main__':
    MiApp().run()


class AnimationScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.label = Label(text='BIENVENIDO A MAFURCOM', font_name='Georgia', font_size='55sp', opacity=0, color=(0.0, 0.0, 1.0, 1.0))
        self.layout.add_widget(self.label)
        self.add_widget(self.layout)

    def on_enter(self):
        anim = Animation(opacity=3, duration=4) + Animation(opacity=1, duration=1) + Animation(opacity=0, duration=2)
        anim.bind(on_complete=self.switch_to_register)
        anim.start(self.label)

    def switch_to_register(self, *args):
        self.manager.current = 'register'

class RegisterScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.name_input = TextInput(multiline=False, hint_text='Ingrese su nombre', font_name='Georgia', font_size='20sp')
        register_button = Button(text='Registrarse',font_name='Georgia', font_size='20sp', on_press=self.register)
        layout.add_widget(Label(text='Registro', font_name='Georgia', font_size='20sp'))
        layout.add_widget(self.name_input)
        layout.add_widget(register_button)
        self.add_widget(layout)

    def register(self, instance):
        app = App.get_running_app()
        app.username = self.name_input.text
        self.manager.current = 'main'

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        layout.add_widget(Label(text='Menú Principal'))
        layout.add_widget(Button(text='Aportes', on_press=self.go_to_upload))
        layout.add_widget(Button(text='Ver Niveles', on_press=self.go_to_levels))
        layout.add_widget(Button(text='LSA Mantenimiento', on_press=self.go_to_lsa))
        layout.add_widget(Button(text='Derechos de Autor', on_press=self.go_to_copyright))
        self.add_widget(layout)

    def go_to_upload(self, instance):
        self.manager.current = 'upload'

    def go_to_levels(self, instance):
        self.manager.current = 'levels'

    def go_to_lsa(self, instance):
        self.manager.current = 'lsa'

    def go_to_copyright(self, instance):
        self.manager.current = 'copyright'

class UploadScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.file_chooser = FileChooserListView()
        self.level_input = TextInput(multiline=False, hint_text='Número de nivel (1-20)')
        upload_button = Button(text='Subir', on_press=self.upload_file)
        back_button = Button(text='Volver', on_press=self.go_back)
        layout.add_widget(self.file_chooser)
        layout.add_widget(self.level_input)
        layout.add_widget(upload_button)
        layout.add_widget(back_button)
        self.add_widget(layout)

    def upload_file(self, instance):
        app = App.get_running_app()
        if self.file_chooser.selection and self.level_input.text:
            file_path = self.file_chooser.selection[0]
            level = int(self.level_input.text)
            if 1 <= level <= 20:
                file_name = os.path.basename(file_path)
                if file_name.lower().endswith(('.mp4', '.avi', '.mov')):
                    if len(app.videos) < 10:
                        app.videos.append((level, file_path))
                        print(f"Video añadido al nivel {level}")
                    else:
                        print("Límite de videos alcanzado")
                else:
                    if len(app.documents) < 5:
                        app.documents.append((level, file_path))
                        print(f"Documento añadido al nivel {level}")
                    else:
                        print("Límite de documentos alcanzado")
            else:
                print("Nivel inválido")

    def go_back(self, instance):
        self.manager.current = 'main'

class LevelsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        scroll_view = ScrollView()
        self.buttons_layout = BoxLayout(orientation='vertical', size_hint_y=None)
        self.buttons_layout.bind(minimum_height=self.buttons_layout.setter('height'))
        scroll_view.add_widget(self.buttons_layout)
        self.layout.add_widget(scroll_view)
        self.layout.add_widget(Button(text='Volver', on_press=self.go_back, size_hint_y=None, height=50))
        self.add_widget(self.layout)

    def on_pre_enter(self):
        self.buttons_layout.clear_widgets()
        for i in range(1, 21):
            btn = Button(text=f'Nivel {i}', size_hint_y=None, height=50)
            btn.bind(on_press=lambda x, level=i: self.show_level_content(level))
            self.buttons_layout.add_widget(btn)

    def show_level_content(self, level):
        app = App.get_running_app()
        content_screen = ContentScreen(name=f'content_{level}')
        content_screen.load_content(level, app.videos, app.documents)
        self.manager.add_widget(content_screen)
        self.manager.current = f'content_{level}'

    def go_back(self, instance):
        self.manager.current = 'main'

class ContentScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.add_widget(self.layout)

    def load_content(self, level, videos, documents):
        self.layout.clear_widgets()
        self.layout.add_widget(Label(text=f'Contenido del Nivel {level}'))
        
        level_videos = [v for l, v in videos if l == level]
        level_documents = [d for l, d in documents if l == level]
        
        if level_videos:
            video_player = Video(source=level_videos[0], state='play', options={'eos': 'loop'})
            self.layout.add_widget(video_player)
        
        for doc in level_documents:
            self.layout.add_widget(Label(text=f'Documento: {os.path.basename(doc)}'))
        
        self.layout.add_widget(Button(text='Volver a Niveles', on_press=self.go_back))

    def go_back(self, instance):
        self.manager.current = 'levels'

class LSAScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.text_input = TextInput(multiline=True, hint_text='Pregúntame algo sobre mantenimiento')
        self.response_label = Label(text='', font_size=20)
        layout.add_widget(Label(text='LSA Mantenimiento'))
        layout.add_widget(self.text_input)
        layout.add_widget(Button(text='Preguntar', on_press=self.ask_lsa))
        layout.add_widget(self.response_label)
        layout.add_widget(Button(text='Volver', on_press=self.go_back))
        self.add_widget(layout)

    def ask_lsa(self, instance):
        pregunta = self.text_input.text
        respuesta = self.get_response(pregunta)
        self.response_label.text = respuesta

    def get_response(self, pregunta):
        import nltk
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        tokens = word_tokenize(pregunta)
        stop_words = set(stopwords.words('spanish'))
        tokens = [t for t in tokens if t.lower() not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        patterns = {
            '¿qué es mantenimiento?': 'El mantenimiento es el conjunto de actividades destinadas a conservar y reparar equipos y sistemas.',
            '¿por qué es importante el mantenimiento?': 'El mantenimiento es importante porque ayuda a prevenir fallas, reducir costos y aumentar la eficiencia.',
            # ...
        }

        respuesta = ''
        for patron, respuesta_patron in patterns.items():
            patron_tokens = word_tokenize(patron)
            if set(patron_tokens).issubset(set(tokens)):
                respuesta = respuesta_patron
                break

        if not respuesta:
            respuesta = 'Lo siento, no tengo una respuesta para esa pregunta.'

        return respuesta

    def go_back(self, instance):
        self.manager.current = 'main'

    import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spacy import Spanish
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer()
modelo = None

def entrenar_modelo(preguntas, respuestas):
    global modelo
    X = vectorizer.fit_transform(preguntas)
    y = respuestas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    modelo = cosine_similarity(X_train, X_test)
    return modelo

nlp = Spanish()

def procesar_pregunta(pregunta):
    tokens = word_tokenize(pregunta)
    tokens = [t for t in tokens if t not in stopwords.words('spanish')]
    doc = nlp(' '.join(tokens))
    return doc

class Interfaz(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.text_input = TextInput(multiline=False)
        self.add_widget(self.text_input)
        self.button = Button(text='Preguntar')
        self.button.bind(on_press=self.preguntar)
        self.add_widget(self.button)

    def preguntar(self, instance):
        pregunta = self.text_input.text
        respuesta = self.get_respuesta(pregunta)
        print(respuesta)

    def get_respuesta(self, pregunta):
        # Llamar al modelo de NLP y aprendizaje automático
        pass

import sqlite3

conn = sqlite3.connect('base_conocimiento.db')
cursor = conn.cursor()

def crear_tabla():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS preguntas (
            id INTEGER PRIMARY KEY,
            pregunta TEXT,
            respuesta TEXT
        )
    ''')
    conn.commit()

def agregar_pregunta(pregunta, respuesta):
    cursor.execute('INSERT INTO preguntas (pregunta, respuesta) VALUES (?, ?)', (pregunta, respuesta))
    conn.commit()

def obtener_respuesta(pregunta):
    cursor.execute('SELECT respuesta FROM preguntas WHERE pregunta = ?', (pregunta,))
    respuesta = cursor.fetchone()
    return respuesta[0] if respuesta else None

from interfaz import AppInterfaz
from nlp import procesar_pregunta
from modelo import entrenar_modelo
from base_conocimiento import crear_tabla, agregar_pregunta, obtener_respuesta

def main():
    crear_tabla()
    # Cargar preguntas y respuestas desde la base de conocimiento
    preguntas = []
    respuestas = []
    # Entrenar modelo
    modelo = entrenar_modelo(preguntas, respuestas)
    # Iniciar interfaz
    AppInterfaz().run()

if __name__ == '__main__':
    main()

class AppInterfaz(App):
    def build(self):
        return Interfaz()

if __name__ == '__main__':
    AppInterfaz().run()

class CopyrightScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.copyright_input = TextInput(multiline=True, hint_text='ASTRID JULIANA ACEVEDO VARGAS  EDWIN SANTIAGO OJEDA SILVA  DORA LORENA SUÁREZ PÉREZ', font_name='Georgia', font_size='40sp')
        back_button = Button(text='Volver', on_press=self.go_back)
        layout.add_widget(Label(text='Derechos de Autor'))
        layout.add_widget(self.copyright_input)
        layout.add_widget(back_button)
        self.add_widget(layout)

    def go_back(self, instance):
        self.manager.current = 'main'

class FinalScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')
        self.label = Label(text='', font_size='20sp', opacity=0)
        self.layout.add_widget(self.label)
        self.add_widget(self.layout)

    def on_enter(self):
        app = App.get_running_app()
        self.label.text = f"¡Gracias por usar la App Educativa!\n\n{app.copyright_text}"
        anim = Animation(opacity=1, duration=2) + Animation(opacity=1, duration=3) + Animation(opacity=0, duration=2)
        anim.bind(on_complete=self.exit_app)
        anim.start(self.label)

    def exit_app(self, *args):
        App.get_running_app().stop()

class EducationalApp(App):
    def build(self):
        self.username = ""
        self.videos = []
        self.documents = []
        self.copyright_text = ""
        
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(AnimationScreen(name='animation'))
        sm.add_widget(RegisterScreen(name='register'))
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(UploadScreen(name='upload'))
        sm.add_widget(LevelsScreen(name='levels'))
        sm.add_widget(LSAScreen(name='lsa'))
        sm.add_widget(CopyrightScreen(name='copyright'))
        sm.add_widget(FinalScreen(name='final'))
        
        return sm

if __name__ == '__main__':
    EducationalApp().run() 
