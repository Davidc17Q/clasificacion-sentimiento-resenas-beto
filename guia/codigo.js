/* Pestaña "Código": explicación bloque por bloque. El texto de los archivos lo inyecta construir_guia.py en D.codigo;
   aquí solo van los rangos de líneas y la explicación de cada bloque. */
const ARCHIVOS = [
 {f:'config.py', rol:'Las decisiones fijas del proyecto', correr:'No se corre solo: los demás archivos lo importan.', bloques:[
  {r:[1,9], t:'Rutas de carpetas', q:'Define dónde están los datos crudos, los procesados, las reseñas de Colombia, los resultados y los modelos. <code>Path(__file__)...parent.parent</code> es “la carpeta raíz del proyecto”, calculada a partir de dónde está este archivo.', s:'Es la libreta de direcciones: así ningún otro archivo tiene rutas escritas a mano.'},
  {r:[11,11], t:'La semilla', q:'<code>SEMILLA = 42</code> fija el punto de partida de todo lo que es “al azar”: qué reseñas se eligen, cómo se dividen, los pesos iniciales de la capa nueva, el orden de los lotes.', s:'Con la misma semilla, el azar sale igual cada vez. Así cualquiera que corra el código obtiene los mismos resultados (reproducibilidad). El 42 no tiene nada especial; es una costumbre.'},
  {r:[13,17], t:'Las tres clases', q:'<code>ETIQUETAS</code> es la lista de nombres. <code>ID2LABEL</code> y <code>LABEL2ID</code> traducen entre número y nombre (0↔negativo, 1↔neutral, 2↔positivo). La red trabaja con números; los nombres son para los humanos.', s:'Un diccionario de traducción entre lo que entiende la red (0, 1, 2) y lo que entendemos nosotros.'},
  {r:[20,25], t:'De estrellas a sentimiento', q:'La regla del proyecto: 1–2 estrellas → 0 (negativo), 3 → 1 (neutral), 4–5 → 2 (positivo). Así salen las “respuestas correctas” para entrenar sin etiquetar a mano 208.000 reseñas.', s:'Las estrellas que puso cada cliente nos dicen la respuesta correcta.'},
  {r:[28,29], t:'Tamaño del corpus balanceado', q:'10.000 reseñas por clase. La clase negativa es la que menos tiene (~10.900), por eso el tope.', s:'Tomamos la misma cantidad de cada tipo para que la red no se sesgue hacia lo positivo.'},
  {r:[31,34], t:'Los modelos de Hugging Face', q:'Nombres oficiales de los modelos preentrenados que se descargan: BETO (<code>dccuchile/bert-base-spanish-wwm-cased</code>) y DistilBERT multilingüe.', s:'La dirección desde donde se descarga cada red ya entrenada.'},
 ]},
 {f:'preparar_datos.py', rol:'Etapa 1: limpiar los datos y dividirlos', correr:'python preparar_datos.py', bloques:[
  {r:[1,18], t:'Descripción e importaciones', q:'El comentario inicial resume la etapa. Se importan <code>ftfy</code> (arregla texto mal codificado), <code>pandas</code> (tablas) y <code>train_test_split</code> de scikit-learn (dividir datos).', s:'La lista de herramientas que se van a usar.'},
  {r:[21,23], t:'Limpiar un texto', q:'<code>ftfy.fix_text</code> repara el “mojibake”: tildes dañadas como “CancÃºn” → “Cancún”. Luego <code>" ".join(texto.split())</code> deja un solo espacio entre palabras y quita saltos de línea.', s:'Arregla letras dañadas y espacios de más. No quita tildes ni mayúsculas, porque le dan información a la red.'},
  {r:[27,36], t:'Cargar, unir y etiquetar', q:'Lee el CSV de Rest-Mex. Une título + reseña en una sola columna <code>texto</code>. Descarta textos de 10 caracteres o menos. <b>Quita duplicados</b> (<code>drop_duplicates</code>) antes de dividir, para que una misma reseña no quede en entrenamiento y en prueba a la vez. Convierte estrellas en clase con <code>estrellas_a_clase</code>.', s:'Deja una tabla limpia: cada fila es una reseña, con su texto y su respuesta correcta.'},
  {r:[38,38], t:'Distribución real', q:'Calcula qué porcentaje de reseñas es de cada clase en la vida real (87% positivas). Se usa después para el “test natural”.', s:'Anota cuántas reseñas de cada tipo hay en la realidad.'},
  {r:[40,43], t:'Balancear', q:'Agrupa por clase y toma 10.000 al azar de cada una (<code>sample</code> con la semilla). Luego las mezcla. <code>estrato</code> combina clase + tipo de negocio (hotel, restaurante, atractivo) para la división siguiente.', s:'Iguala las tres clases para que la red aprenda a reconocerlas todas.'},
  {r:[45,47], t:'Dividir 70 / 15 / 15', q:'Primero separa 70% para entrenar y 30% aparte; luego parte ese 30% por la mitad: 15% validación y 15% prueba. <code>stratify</code> hace que los tres grupos tengan las mismas proporciones de clase y tipo de negocio.', s:'Entrenamiento = estudiar. Validación = simulacros para elegir ajustes. Prueba = examen final que solo se mira al final.'},
  {r:[49,56], t:'Prueba con distribución natural', q:'Con reseñas que <b>no</b> entraron al corpus balanceado (<code>df.drop(index=bal.index)</code>), arma 10.000 con la proporción real (87% positivas).', s:'Un segundo examen “como en la vida real”, para ver cómo se comporta fuera del laboratorio balanceado.'},
  {r:[58,74], t:'Guardar', q:'Escribe <code>train.csv</code>, <code>val.csv</code>, <code>test.csv</code> y <code>test_natural.csv</code>, más un <code>resumen_datos.json</code> con tamaños, distribuciones y longitudes (de ahí salen los números de la pestaña Datos).', s:'Guarda los cuatro grupos para que los demás archivos los usen.'},
 ]},
 {f:'metricas.py', rol:'Cómo se califica a todos los modelos', correr:'No se corre solo: lo usan los demás.', bloques:[
  {r:[8,20], t:'Calcular las métricas', q:'Recibe las respuestas correctas y las predichas, y devuelve: <b>accuracy</b> (porcentaje de aciertos), <b>F1 macro</b> (promedio del F1 de las 3 clases, sin ponderar), precisión / recall / F1 por clase y la <b>matriz de confusión</b> (qué se confunde con qué). Todos los modelos usan esta misma función, así la comparación es justa.', s:'La regla única para poner nota. F1 macro le da el mismo peso a las tres clases, aunque una sea más común.'},
 ]},
 {f:'baseline_tfidf.py', rol:'Etapa 2a: el modelo clásico de comparación', correr:'python baseline_tfidf.py  (tarda segundos: se puede correr en vivo)', bloques:[
  {r:[15,22], t:'Armar y entrenar el modelo', q:'<code>make_pipeline</code> encadena dos pasos. <b>TF-IDF</b> convierte cada reseña en números: cuenta palabras y pares de palabras (<code>ngram_range=(1,2)</code>), ignora las que aparecen en menos de 2 reseñas (<code>min_df=2</code>) y le da más peso a las palabras raras. Luego una <b>regresión logística</b> aprende un peso por palabra para cada clase. <code>fit</code> entrena.', s:'El método “antiguo”: cuenta palabras sin entender el orden ni el contexto. Sirve para demostrar que BETO sí aporta.'},
  {r:[23,30], t:'Evaluar', q:'Predice en validación, prueba, prueba natural y Colombia, y califica con <code>metricas.calcular</code>.', s:'Le pone nota en los mismos exámenes que a BETO.'},
  {r:[32,38], t:'Palabras más influyentes', q:'Para cada clase, toma las 15 palabras con mayor peso (<code>coef_</code>). Es una forma de ver “qué aprendió”: en negativo salen cosas como “pésimo”, “sucio”.', s:'Muestra qué palabras empujan a cada clase.'},
  {r:[40,44], t:'Guardar y devolver', q:'Escribe <code>resultados/tfidf.json</code> y devuelve el modelo (lo reutiliza <code>evaluar_externo.py</code>).', s:'Guarda la nota.'},
 ]},
 {f:'entrenar_transformer.py', rol:'Etapa 2b: el corazón, el ajuste fino de BETO', correr:'python entrenar_transformer.py --modelo beto --lr 2e-5 --max_len 256', bloques:[
  {r:[36,45], t:'Opciones desde la terminal', q:'<code>argparse</code> permite cambiar el modelo, la tasa de aprendizaje (<code>--lr</code>), las épocas, la longitud máxima y el tamaño de lote sin tocar el código. Así se hizo la búsqueda de hiperparámetros.', s:'Las perillas que se pueden mover al lanzar el entrenamiento.'},
  {r:[49,54], t:'Semilla y carpeta de salida', q:'<code>set_seed(SEMILLA)</code> fija el azar de Python, NumPy y PyTorch. El nombre de la corrida sale de los hiperparámetros (p. ej. <code>beto_lr2e-05_len256</code>).', s:'Garantiza resultados repetibles y le pone nombre a cada experimento.'},
  {r:[56,60], t:'Cargar BETO y agregar la capa nueva', q:'<code>AutoTokenizer</code> carga el tokenizador de BETO (cómo partir el texto en tokens). <code>AutoModelForSequenceClassification</code> carga BETO preentrenado <b>y le agrega encima una capa lineal nueva</b> de 768 → 3 (<code>num_labels=3</code>). Esa capa empieza al azar; todo lo demás ya sabe español.', s:'Aquí está la arquitectura: BETO completo + una capa nueva que decide entre 3 clases. No se escriben las 12 capas a mano porque se reutilizan (transfer learning).'},
  {r:[62,68], t:'Convertir texto en tokens', q:'<code>a_dataset</code> pasa cada tabla a un <code>Dataset</code> de Hugging Face y la tokeniza: cada reseña se vuelve una lista de números (<code>input_ids</code>) más la máscara de atención. <code>truncation=True, max_length</code> corta las reseñas largas.', s:'Traduce cada reseña a los números que la red puede leer.'},
  {r:[70,72], t:'¿Cuántas reseñas se cortan?', q:'Mide qué fracción de reseñas llega al límite de tokens. Con 128 tokens se cortaba el 25% de las reseñas; con 256, solo el 2%. Eso justifica la elección de 256.', s:'Comprueba cuánta información se pierde al cortar.'},
  {r:[74,77], t:'Cómo se califica durante el entrenamiento', q:'La red da 3 logits por reseña; <code>argmax</code> elige el mayor como predicción. Luego se calcula accuracy y F1 macro en validación.', s:'Cada cierto tiempo, se le hace un simulacro y se anota la nota.'},
  {r:[79,101], t:'Los hiperparámetros', q:'<ul class="limpia"><li><code>learning_rate</code>: tamaño del paso de AdamW (2e-05 = 0,00002).</li><li><code>num_train_epochs</code>: máximo de pasadas por los datos.</li><li><code>per_device_train_batch_size</code>: reseñas por lote (32).</li><li><code>weight_decay=0.01</code>: la “W” de AdamW; encoge un poco los pesos para no memorizar.</li><li><code>warmup_steps=0.1</code>: el primer 10% de los pasos la tasa sube desde ~0; luego baja en línea recta (<code>lr_scheduler_type="linear"</code>).</li><li><code>eval_steps=200</code>, <code>save_steps=200</code>: cada 200 pasos evalúa en validación y guarda una copia.</li><li><code>load_best_model_at_end</code> + <code>metric_for_best_model="f1_macro"</code>: al final se queda con la copia de mejor F1 en validación, no con la última.</li><li><code>fp16</code>: cálculos en media precisión en la GPU (más rápido, menos memoria).</li></ul>', s:'Todas las perillas del entrenamiento, en un solo lugar.'},
  {r:[102,111], t:'El entrenador', q:'<code>Trainer</code> hace el ciclo completo: lote → predicción → pérdida (entropía cruzada) → gradientes (backpropagation) → paso de AdamW. <code>DataCollatorWithPadding</code> rellena cada lote hasta la reseña más larga de ese lote. <code>EarlyStoppingCallback(4)</code> detiene si el F1 en validación no mejora en 4 evaluaciones seguidas.', s:'El “profesor” automático que repite: mira ejemplos, se equivoca, corrige. Y para solo si deja de mejorar.'},
  {r:[113,115], t:'Entrenar', q:'<code>trainer.train()</code> ejecuta todo el entrenamiento y se mide cuánto tardó.', s:'Aquí la red aprende.'},
  {r:[117,127], t:'Anotar la ficha técnica', q:'Guarda hiperparámetros, número de parámetros entrenables (~110 millones: <b>se ajustan todos</b>, no se congela nada), fracción truncada, tiempo, mejor checkpoint y GPU usada.', s:'La ficha de cada experimento.'},
  {r:[129,138], t:'Examen final', q:'Predice en validación, prueba y prueba natural. Para la prueba aplica <code>softmax</code> a los logits y guarda las probabilidades por reseña en <code>predicciones_test.csv</code> (de ahí salen los errores que muestra la guía).', s:'Ahora sí se abre el examen final y se guarda la respuesta de cada reseña.'},
  {r:[140,144], t:'Reseñas de Colombia', q:'Si existe el conjunto colombiano, lo tokeniza y lo evalúa igual.', s:'Un examen extra con reseñas reales de aquí.'},
  {r:[146,152], t:'Guardar', q:'Escribe <code>metricas.json</code> y <code>historial.json</code> (pérdida y F1 a lo largo del entrenamiento: de ahí salen las curvas). Opcionalmente guarda el modelo.', s:'Guarda las notas y las curvas de aprendizaje.'},
 ]},
 {f:'correr_busqueda.sh', rol:'La búsqueda de hiperparámetros', correr:'bash correr_busqueda.sh', bloques:[
  {r:[1,12], t:'Probar varias tasas de aprendizaje', q:'Entrena BETO con tres tasas (3e-5, 2e-5 y 5e-5) y luego DistilBERT multilingüe, cada uno con su registro en <code>logs/</code>. Después se corrió aparte la mejor (2e-5) con <code>max_len 256</code>, que fue la elegida por F1 en validación.', s:'Prueba varias configuraciones para escoger la mejor con los simulacros, no con el examen final.'},
 ]},
 {f:'analisis_capas.py', rol:'Etapa 3b: las etapas intermedias', correr:'python analisis_capas.py --modelo_ft <mejor checkpoint>', bloques:[
  {r:[27,34], t:'Configuración', q:'700 reseñas por clase de la prueba (2.100). Las capas que se dibujan con PCA (la 0 no, porque ahí el [CLS] es igual para todas las reseñas). Cuatro frases para ver la atención.', s:'Qué muestras se van a analizar.'},
  {r:[37,45], t:'Sacar el [CLS] de cada capa', q:'<code>output_hidden_states=True</code> pide las 13 salidas internas (embeddings + 12 capas), cada una de forma [lote, tokens, 768]. <code>h[:, 0, :]</code> toma la posición 0, que es el [CLS]. Resultado: para cada reseña, 13 vectores de 768 números. <code>@torch.no_grad()</code>: no se entrena, solo se mira.', s:'Le toma una “foto” al resumen [CLS] después de cada etapa.'},
  {r:[48,57], t:'Atención del [CLS]', q:'En la última capa, promedia las 12 cabezas de atención y toma la fila del [CLS]: cuánto se fija el [CLS] en cada token. Eso es lo que la guía colorea.', s:'Muestra en qué palabras se fijó el resumen para decidir.'},
  {r:[60,67], t:'Sondas lineales', q:'Para cada una de las 13 capas, entrena una regresión logística con el vector [CLS] de esa capa (70%) y mide F1 en el 30% restante. Si una sonda tan simple clasifica bien, esa capa “ya contiene” el sentimiento.', s:'Un mini-examen a cada capa: ¿con lo que sabes hasta aquí, ya puedes decir el sentimiento?'},
  {r:[70,77], t:'PCA en 2D', q:'Reduce los 768 números a 2 para poder dibujarlos. Si los puntos de cada clase quedan en grupos separados, la capa los distingue.', s:'Aplasta 768 dimensiones en un plano para verlas con los ojos.'},
  {r:[80,101], t:'Comparar antes y después', q:'Hace todo para BETO <b>preentrenado</b> (sin ajustar) y para BETO <b>ajustado</b>, y guarda <code>capas.json</code>. Resultado: el preentrenado ya distingue bastante desde las capas tempranas (pico de 0,66 en la capa 5) y luego baja; el ajustado sigue subiendo hasta 0,75 en la capa 12. El ajuste fino especializó las últimas capas en sentimiento.', s:'Muestra qué aportó el ajuste fino capa por capa.'},
 ]},
 {f:'evaluar_externo.py', rol:'Etapa 3c: la prueba en Colombia', correr:'python evaluar_externo.py', bloques:[
  {r:[23,33], t:'Predecir con un modelo guardado', q:'<code>pipeline("text-classification")</code> carga un checkpoint y devuelve, con <code>top_k=None</code>, la probabilidad de las 3 clases por reseña. Se elige la mayor.', s:'Pasa cada reseña colombiana por el modelo.'},
  {r:[42,71], t:'BETO y DistilBERT en Colombia', q:'Evalúa los dos modelos. Para BETO además calcula el acierto por tipo de negocio y por ciudad, y cuánto coincide la etiqueta manual con la regla de estrellas.', s:'La nota en Colombia, desglosada.'},
  {r:[73,77], t:'TF-IDF en Colombia', q:'Reentrena el baseline (tarda segundos) y predice las mismas reseñas.', s:'El modelo clásico también presenta el examen colombiano.'},
  {r:[83,106], t:'Bootstrap: ¿la diferencia es real?', q:'Repite 2.000 veces: toma 166 reseñas al azar con reemplazo y recalcula el F1 de cada modelo. El rango donde cae el 95% de los resultados es el intervalo de confianza. Si el intervalo de la diferencia BETO − otro incluye el 0, no se puede decir que uno sea mejor.', s:'Con pocas reseñas, la suerte pesa. Esto mide cuánto.'},
 ]},
 {f:'demo.py', rol:'La prueba de funcionamiento en vivo', correr:'python demo.py --ejemplos', bloques:[
  {r:[17,23], t:'Silenciar avisos', q:'Oculta mensajes técnicos y barras de progreso para que la demo se vea limpia.', s:'Que en pantalla solo salga lo importante.'},
  {r:[28,35], t:'Frases preparadas', q:'Seis reseñas pensadas para mostrar: queja clara, elogio claro, neutral, mixta, sarcasmo y una muy corta.', s:'Ejemplos listos para lucirse.'},
  {r:[41,54], t:'Cargar el mejor BETO', q:'Lee qué corrida ganó (<code>beto_final.json</code>), busca su mejor checkpoint y lo carga en la GPU si hay. <code>.eval()</code> apaga el dropout.', s:'Abre la red ya entrenada que quedó guardada en el PC.'},
  {r:[57,61], t:'Clasificar', q:'Tokeniza el texto, lo pasa por el modelo, toma los 3 logits y aplica <code>softmax</code> para tener probabilidades.', s:'Texto → tokens → BETO → 3 puntajes → porcentajes.'},
  {r:[64,74], t:'Mostrar', q:'Imprime la clase ganadora, cuántos tokens tuvo, cuánto tardó y una barra por clase.', s:'Lo que ve el profe en pantalla.'},
  {r:[77,103], t:'Modo interactivo', q:'Si se pasa un texto, clasifica solo ese. Con <code>--ejemplos</code> corre las frases preparadas. Luego espera reseñas escritas hasta un Enter vacío o “salir”.', s:'Escribes una reseña y responde al instante.'},
 ]},
];

const PREG_CODIGO = [
 ['¿Dónde está la arquitectura? No veo las capas.', 'En <code>entrenar_transformer.py</code>, líneas 58–60: <code>AutoModelForSequenceClassification.from_pretrained(..., num_labels=3)</code> carga BETO completo (12 capas) y le agrega la capa lineal de 3 salidas. No se escriben a mano porque se reutiliza un modelo preentrenado. Si piden verla: <code>print(modelo)</code> muestra todas las capas.'],
 ['¿Dónde está la función de pérdida? ¿Y la softmax?', 'Dentro del modelo de Hugging Face: como le pasamos la columna <code>labels</code>, calcula automáticamente la entropía cruzada, que incluye la softmax. La softmax explícita aparece al guardar probabilidades (línea 134) y en <code>demo.py</code>.'],
 ['¿Dónde está AdamW?', 'Es el optimizador por defecto del <code>Trainer</code>. En <code>TrainingArguments</code> le damos su tasa de aprendizaje (<code>learning_rate</code>) y su weight decay (<code>weight_decay=0.01</code>).'],
 ['¿Dónde está el backpropagation?', 'Dentro de <code>trainer.train()</code>: en cada lote PyTorch calcula la pérdida y llama a <code>backward()</code>, que obtiene los gradientes de los ~110 millones de pesos; luego AdamW da el paso.'],
 ['¿Cómo evitas que el modelo vea el examen (fuga de datos)?', 'Los duplicados se quitan antes de dividir (<code>preparar_datos.py</code> línea 33), la división se hace una sola vez con semilla fija, y la prueba nunca se usa para elegir hiperparámetros: eso se hace con validación.'],
 ['¿Qué hace <code>stratify</code>?', 'Que entrenamiento, validación y prueba tengan las mismas proporciones de clase y de tipo de negocio.'],
 ['¿Qué es <code>argmax</code>?', 'Devuelve la posición del número más grande. Entre los 3 logits, la posición ganadora es la clase predicha.'],
 ['¿Por qué <code>@torch.no_grad()</code>?', 'Porque en el análisis y en la demo no se entrena: no hace falta calcular gradientes, y así va más rápido y usa menos memoria.'],
 ['¿Qué es <code>DataCollatorWithPadding</code>?', 'Las reseñas tienen distinto largo; para procesarlas juntas en un lote se rellenan con tokens vacíos hasta la más larga de ese lote. La máscara de atención le dice a la red que ignore ese relleno.'],
 ['¿Qué es un checkpoint?', 'Una copia de todos los pesos en un momento del entrenamiento. Se guardó uno cada 200 pasos y se conservó el de mejor F1 en validación (el paso 1.200 en la corrida ganadora).'],
 ['¿Puedes correr algo en vivo?', '<code>python baseline_tfidf.py</code> entrena en segundos. Y <code>python demo.py</code> clasifica cualquier reseña con el BETO entrenado. Entrenar BETO desde cero no se hace en vivo porque tarda.'],
];

function bloqueCodigo(arch, b){
  const lineas = (D.codigo && D.codigo[arch.f] || '').split('\n');
  const trozo = lineas.slice(b.r[0]-1, b.r[1]).map((l,i)=>`<span class="ln">${b.r[0]+i}</span>${esc(l)}`).join('\n');
  return `<article class="bloque-codigo"><div class="cab"><h3>${b.t}</h3><span class="pill">líneas ${b.r[0]}${b.r[1]>b.r[0]?'–'+b.r[1]:''}</span></div>
   <pre class="codigo"><code>${trozo}</code></pre>
   <div class="explica"><span class="eyebrow">Qué hace</span><div>${b.q}</div></div>
   <div class="simple"><b class="et">En palabras simples</b><p>${b.s}</p></div></article>`;
}
function esc(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function pintarArchivo(i){
  const a = ARCHIVOS[i];
  document.querySelectorAll('#ctrl-arch button').forEach(b=>b.setAttribute('aria-pressed', b.dataset.i==i));
  $('#arch').innerHTML = `<div class="bloque"><span class="eyebrow">Archivo ${i+1} de ${ARCHIVOS.length}</span><h2><code>${a.f}</code></h2><p class="lead">${a.rol}</p>
   <div class="formula">${esc(a.correr)}</div></div>${a.bloques.map(b=>bloqueCodigo(a,b)).join('')}
   ${i<ARCHIVOS.length-1?`<button class="siguiente" data-sig="${i+1}">Siguiente archivo: ${ARCHIVOS[i+1].f} →</button>`:''}`;
  const s = document.querySelector('[data-sig]'); if(s) s.onclick=()=>{pintarArchivo(+s.dataset.sig); $('#arch').scrollIntoView({block:'start'});};
}
function vistaCodigo(){ return `
<section class="vista">
  <div class="bloque"><span class="eyebrow">Revisión del código</span><h1>El código, bloque por bloque</h1>
  <p class="lead">Los archivos están en el orden del proceso: decisiones fijas → datos → modelo clásico → BETO → búsqueda → etapas intermedias → Colombia → demo. Cada bloque trae el código real, qué hace y la versión en palabras simples.</p>
  <div class="panel"><p><b>Cómo mostrarlo:</b> abre el repositorio en VS Code y recorre los archivos en este orden. En cada uno di primero su frase de “rol” y luego señala los bloques. Termina con <code>resultados/</code> (los JSON con las métricas) y la demo en vivo.</p></div></div>
  <div class="control" id="ctrl-arch">${ARCHIVOS.map((a,i)=>`<button data-i="${i}">${a.f}</button>`).join('')}</div>
  <div id="arch" class="bloque" style="scroll-margin-top:70px"></div>
  <div class="bloque"><h2>Preguntas sobre el código</h2>
  <div class="lista-preguntas">${PREG_CODIGO.map(([p,r])=>`<details class="pregunta"><summary>${p}</summary><div class="resp"><p>${r}</p></div></details>`).join('')}</div></div>
  <div class="aviso"><b>Si preguntan “¿esto lo hiciste tú?”</b><p>Sé honesto: si usaste un asistente de IA para escribir parte del código, dilo. Lo que importa es poder explicar cualquier línea que te señalen. Esta pestaña existe para eso.</p></div>
</section>`;}
