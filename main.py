# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from typing import List
import uvicorn

#import nest_asyncio
from fastapi import FastAPI, Query, HTTPException
import httpx
import pickle
from joblib import load
import numpy as np
from fastapi.responses import HTMLResponse
import pandas as pd
import os

# Применяем nest_asyncio для работы в Jupyter
#nest_asyncio.apply()


app = FastAPI(title="Стадия Эмоционального Выгорания: AI детектор")

# Храним активные подключения
active_connections = []

from pathlib import Path



import pickle

with open('models/model_napr.pkl', 'rb') as f:
    # Пытаемся загрузить без выполнения
    try:
        data = pickle.load(f)
        print(f"✅ Модель загружена: {type(data)}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        
    # Попробуем прочитать метаданные
    f.seek(0)
    content = f.read(1000)  # первые 1000 байт
    if b'scikit-learn' in content:
        print("Модель содержит scikit-learn")


# Загружаем модели
MODEL_PATH_NAPR = "models/model_napr.pkl"
# Диагностика: проверьте существование файла
print(f"Текущая рабочая директория: {os.getcwd()}")
print(f"Путь к модели: {MODEL_PATH_NAPR}")
print(f"Абсолютный путь: {os.path.abspath(MODEL_PATH_NAPR)}")


# Проверка существования файла
if os.path.exists(MODEL_PATH_NAPR):
    print(f"✅ Файл модели существует, размер: {os.path.getsize(MODEL_PATH_NAPR)} байт")
else:
    print(f"❌ Файл модели НЕ существует!")
MODEL_PATH_REZ = "models/model_rez.pkl"
MODEL_PATH_IST = "models/model_ist.pkl"
try:
    with open(MODEL_PATH_NAPR, 'rb') as file:
        model_napr = load(file)
    print("Модель загружена успешно")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model_napr = None

try:
    with open(MODEL_PATH_REZ, 'rb') as file:
        model_rez = load(file)
    print("Модель загружена успешно")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model_rez = None

try:
    with open(MODEL_PATH_IST, 'rb') as file:
        model_ist = load(file)
    print("Модель загружена успешно")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model_ist = None
    
def get_prediction_text_napr(prediction):
    """
    Возвращает текстовое описание в зависимости от значения предсказания
    """
    if prediction < 37:
        return 'У вас низкий риск попасть в фазу Напряжение.'
    elif prediction >=37 and prediction<60:
        return 'У вас средний риск попасть в фазу Напряжение. Можно подумать о повышении эмоциональной насыщенности жизни.'
    else:
        return 'У вас высокий риск попасть в фазу Напряжение. Необходимо поработать над эмоциональной насыщенностью жизни.'
        
def get_prediction_text_rez(prediction):
    """
    Возвращает текстовое описание в зависимости от значения предсказания
    """
    if prediction < 37:
        return 'У вас низкий риск попасть в фазу Резистенция.'
    elif prediction >=37 and prediction<60:
        return 'У вас средний риск попасть в фазу Резистенция. Можно подумать о повышении вовлеченности в Вашу деятельность.'
    else:
        return 'У вас высокий риск попасть в фазу Резистенция. Необходимо поработать над вовлеченностью в Вашу деятельность.'

def get_prediction_text_ist(prediction):
    """
    Возвращает текстовое описание в зависимости от значения предсказания
    """
    if prediction < 37:
        return 'У вас низкий риск попасть в фазу Истощение.'
    elif prediction >=37 and prediction<60:
        return 'У вас средний риск попасть в фазу Истощение. Можно подумать о повышении контроля над происходящим и эмоциональной насыщенности жизни.'
    else:
        return 'У вас высокий риск попасть в фазу Истощение. Необходимо поработать над повышением контроля над происходящим и о повышении эмоциональной насыщенности жизни.'

        
@app.get('/', response_class=HTMLResponse)

async def simple_form():
    return """
    <html>
    <body>
        <h2> Анкета </h2>
        <form action="/calculate">
            Введите ответы на вопросы: <br>
            <br>
            1. Какой у Вас стаж? <br>
            <input type="text" name="field1" value="" style="width: 100px"><br>
            <br>
            Далее Вам будут предложены пары противоположных утверждений. Выберите вариант более точного соответствия вашему состоянию между двумя утверждениями. 1 - максимальное соответствие утверждению слева,  7 - утверждению справа.<br>
            <br>
            2. "Обычно мне очень скучно."    1 2 3 4 5 6 7    "Обычно я полон энергии." <br>
            <input type="text" name="field2" value="" style="width: 100px"><br>
            <br>
            3. "Жизнь кажется мне всегда волнующей и захватывающей."    1 2 3 4 5 6 7    "Жизнь кажется мне совершенно спокойной и рутинной." <br>
            <input type="text" name="field3" value="" style="width: 100px"><br>
            <br>
            4. "Моя жизнь представляется мне крайне бессмысленной и бесцельной."    1 2 3 4 5 6 7    "Моя жизнь представляется мне вполне осмысленной и целеустремленной." <br>
            <input type="text" name="field4" value="" style="width: 100px"><br>
            <br>
            5. "Каждый день кажется мне всегда новым и непохожим на другие."    1 2 3 4 5 6 7    "Каждый день кажется мне совершенно похожим на все другие." <br>
            <input type="text" name="field5" value="" style="width: 100px"><br>
            <br>
            6. "Моя жизнь сложилась именно так, как я мечтал."    1 2 3 4 5 6 7    "Моя жизнь сложилась совсем не так, как я мечтал." <br>
            <input type="text" name="field6" value="" style="width: 100px"><br>
            <br>
            7. "Моя жизнь пуста и неинтересна."    1 2 3 4 5 6 7    "Моя жизнь наполнена интересными делами." <br>
            <input type="text" name="field7" value="" style="width: 100px"><br>
            <br>
            Далее Вам будут предложены утверждения. Дайте один ответ на каждый вопрос, где 0 - нет, 1 - скорее нет, 2 - скорее да, 3 - да. <br>
            <br>
            8. "Иногда мне кажется, что никому нет до меня дела." <br>
            <input type="text" name="field8" value="" style="width: 100px"><br>
            <br>
            9. "Я постоянно занят, и мне это нравится." <br>
            <input type="text" name="field9" value="" style="width: 100px"><br>
            <br>
            10. "Порой все, что я делаю, кажется мне бесполезным."  <br>
            <input type="text" name="field10" value="" style="width: 100px"><br>
            <br>
            11. "Мне кажется, я не живу полной жизнью, а только играю роль."  <br>
            <input type="text" name="field11" value="" style="width: 100px"><br>
            <br>
            12. "Когда кто-­нибудь жалуется, что жизнь скучна, это значит, что он просто не умеет видеть интересное." <br>
            <input type="text" name="field12" value="" style="width: 100px"><br>
            <br>
            13. "Мне всегда есть чем заняться."  <br>
            <input type="text" name="field13" value="" style="width: 100px"><br>
            <br>
            14. "Мне кажется, жизнь проходит мимо меня." <br>
            <input type="text" name="field14" value="" style="width: 100px"><br>
            <br>
            15. "Бывает, жизнь кажется мне скучной и бесцветной."  <br>
            <input type="text" name="field15" value="" style="width: 100px"><br>
            <br>
            16. "Как правило, я работаю с удовольствием." <br>
            <input type="text" name="field16" value="" style="width: 100px"><br>
            <br>
            17. "Иногда я чувствую себя лишним даже в кругу друзей." <br>
            <input type="text" name="field17" value="" style="width: 100px"><br>
            <br>
             18. "Я часто не уверен в собственных решениях." <br>
            <input type="text" name="field18" value="" style="width: 100px"><br>
            <br>
            19. "Вечером я часто чувствую себя совершенно разбитым." <br>
            <input type="text" name="field19" value="" style="width: 100px"><br>
            <br>
            20. "Я всегда уверен, что смогу воплотить в жизнь то, что задумал." <br>
            <input type="text" name="field20" value="" style="width: 100px"><br>
            <br>
            21. "Возникающие проблемы часто кажутся мне неразрешимыми." <br>
            <input type="text" name="field21" value="" style="width: 100px"><br>
            <br>
            22. "Порой мне кажется, что все мои усилия тщетны." <br>
            <input type="text" name="field22" value="" style="width: 100px"><br>
            <br>
            23. "Мне не хватает упорства закончить начатое." <br>
            <input type="text" name="field23" value="" style="width: 100px"><br>
            <br>
            24. "Бывает, на меня наваливается столько проблем, что просто руки опускаются." <br>
            <input type="text" name="field24" value="" style="width: 100px"><br>
            <br>
            25. "Друзья уважают меня за упорство и непреклонность."  <br>
             <input type="text" name="field25" value="" style="width: 100px"><br>
            <br>
            <input type="submit" value="Предсказать фазу ЭВ">
            
        </form>
     
    </body>
    </html>
    """

@app.get('/calculate')
def calculate(field1: str = Query(description='1. Какой у Вас стаж?'),
              field2: str = Query(description='2. "Обычно мне очень скучно."    1 2 3 4 5 6 7    "Обычно я полон энергии."'),
    field3: str = Query(description='3. "Жизнь кажется мне совершенно спокойной и рутинной."    1 2 3 4 5 6 7    "Жизнь кажется мне всегда волнующей и захватывающей."'),
    field4: str = Query(description='4. "Моя жизнь представляется мне крайне бессмысленной и бесцельной."    1 2 3 4 5 6 7    "Моя жизнь представляется мне вполне осмысленной и целеустремленной."'),
    field5: str = Query(description='5. "Каждый день кажется мне всегда новым и непохожим на другие."    1 2 3 4 5 6 7    "Каждый день кажется мне совершенно похожим на все другие."'),
    field6: str = Query(description='6. "Моя жизнь сложилась именно так, как я мечтал."    1 2 3 4 5 6 7    "Моя жизнь сложилась совсем не так, как я мечтал."'),
    field7: str = Query(description='7. "Моя жизнь пуста и неинтересна."    1 2 3 4 5 6 7    "Моя жизнь наполнена интересными делами."'),
    field8: str = Query(description='8. "Иногда мне кажется, что никому нет до меня дела."'),
    field9: str = Query(description='9. "Я постоянно занят, и мне это нравится."'),
    field10: str = Query(description='10. "Порой все, что я делаю, кажется мне бесполезным."'),
    field11: str = Query(description='11. "Мне кажется, я не живу полной жизнью, а только играю роль."'),
    field12: str = Query(description='12. "Когда кто-нибудь жалуется, что жизнь скучна, это значит, что он просто не умеет видеть интересное."'),
    field13: str = Query(description='13. "Мне всегда есть чем заняться."'),
    field14: str = Query(description='14. "Мне кажется, жизнь проходит мимо меня."'),
    field15: str = Query(description='15. "Бывает, жизнь кажется мне скучной и бесцветной."'),
    field16: str = Query(description='16. "Как правило, я работаю с удовольствием."'),
    field17: str = Query(description='17. "Иногда я чувствую себя лишним даже в кругу друзей."'),
    field18: str = Query(description='18. "Я часто не уверен в собственных решениях."'),
    field19: str = Query(description='19. "Вечером я часто чувствую себя совершенно разбитым."'),
    field20: str = Query(description='20. "Я всегда уверен, что смогу воплотить в жизнь то, что задумал."'),
    field21: str = Query(description='21. "Возникающие проблемы часто кажутся мне неразрешимыми." '),
    field22: str = Query(description='22. "Порой мне кажется, что все мои усилия тщетны."'),
    field23: str = Query(description='23. "Мне не хватает упорства закончить начатое."'),
    field24: str = Query(description='24. "Бывает, на меня наваливается столько проблем, что просто руки опускаются."'),
    field25: str = Query(description='25. "Друзья уважают меня за упорство и непреклонность." ')
    ):
    data = f"{field1}"
    try:
        
                    
        feature_1 = float(data) 
        
        #Проверяем количество признаков
        #if len(feature_1) != 1:
        #    return {"error": "Нужно ровно 1 число"}
        
    except ValueError:
        return {"error": "Некорректный формат данных."}
        

    data_1 = f"{field2},{field3},{field4},{field5},{field6},{field7}"

    # Преобразуем строку в список чисел
    try:
                    
        features = [float(x.strip()) for x in data_1.split(',')]
        
        # Проверяем количество признаков
        if len(features) != 6:
            return {"error": "Вы пропустили ответ в вопросах 2-7, пожалуйста заполните все ответы."}
        for value in features:
            if value < 1 or value > 7:
                return {"error": f"Число {value} должно быть в диапазоне от 1 до 7"}
        feature_2=features[0]-features[1]+features[2]-features[3]-features[4]+features[5]+24
        
        
    
    except ValueError:
        return {"error": "Некорректный формат данных."}

    data_2 = f"{field8},{field9},{field10},{field11},{field12},{field13},{field14},{field15},{field16},{field17},{field18},{field19},{field20},{field21},{field22},{field23},{field24},{field25}"

    # Преобразуем строку в список чисел
    try:
                    
        features2 = [float(x.strip()) for x in data_2.split(',')]
        
        # Проверяем количество признаков
        if len(features2) != 18:
            return {"error": "Вы пропустили ответ в вопросах 8-25, пожалуйста заполните все ответы."}
        for value in features2:
            if value < 0 or value > 3:
                return {"error": f"Число {value} должно быть в диапазоне от 0 до 3"}
        feature_3=features2[1]+features2[4]+features2[5]+features2[8]-features2[0]-features2[2]-features2[3]-features2[6]-features2[7]-features2[9]+18
        feature_4=features2[12]+features2[17]-features2[10]-features2[11]-features2[13]-features2[14]-features2[15]-features2[16]+18
    except ValueError:
        return {"error": "Некорректный формат данных."}

    data_array = [feature_1, feature_2, feature_3, feature_4]
    print(data_array)
    prediction_napr= int(model_napr.predict(data_array[1:4]))
    prediction_text_napr = get_prediction_text_napr(prediction_napr)
            
    data_df = pd.DataFrame([data_array[2:4]], columns=['ж_1', 'ж_2'])
    prediction_rez= int(model_rez.predict(data_df)[0])
    prediction_text_rez = get_prediction_text_rez(prediction_rez)

    data_df = pd.DataFrame([data_array], columns=['стаж', 'со_2', 'ж_1', 'ж_2'])
    prediction_ist= int(model_ist.predict(data_df)[0])
    #data_2d = np.array(data_array).reshape(1, -1)
    #prediction_ist= int(model_ist.predict(data_2d)[0])
   
    prediction_text_ist = get_prediction_text_ist(prediction_ist)

     
    return {'cтаж': feature_1,
                'со_2': feature_2,
                'ж_1': feature_3,
                'ж_2': feature_4,
                'Фаза Напряжения': prediction_napr, 'Реко_1': prediction_text_napr, 
                'Фаза Резистенции': prediction_rez, 'Реко_2': prediction_text_rez, 
                'Фаза Истощения': prediction_ist, 'Реко_3': prediction_text_ist}


# WebSocket для реального времени
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Эхо-ответ
            await websocket.send_text(f"Получено: {data}")
    except WebSocketDisconnect:
        active_connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000, reload=True)

#if __name__ == '__main__':
#    config = uvicorn.Config(app=app, host="0.0.0.0", port=8080)
#    server = uvicorn.Server(config)
    
    # Запускаем в текущем event loop
#    import asyncio
#    loop = asyncio.get_event_loop()
#    loop.create_task(server.serve())
#    print("Сервер запущен на http://localhost:8080")
