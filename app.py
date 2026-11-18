import ast
import re

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp
from matplotlib.patches import Circle
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# ==========================================
# 1. CONFIGURACION GENERAL
# ==========================================

st.set_page_config(
    page_title="ALGEBRA MASTER",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

x = sp.Symbol("x")

# Paleta de colores (se reutiliza en CSS y gráficos)
FONDO = "#0D130B"
LIMA = "#B5FF35"
VERDE = "#72B82B"
TEXTO = "#F2F7EC"

ESTILOS = """
<style>
.stApp {
    background: linear-gradient(135deg, #080B08 0%, #111C10 45%, #1B2B14 100%);
    color: #F2F7EC;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #080B08, #152213, #1B2B14);
}
[data-testid="stHeader"] { background: rgba(8, 11, 8, 0.95); }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050805 0%, #14210D 55%, #263D17 100%);
    border-right: 1px solid #8BFF00;
}

.main .stMarkdown, .main label, .main p, .main h1, .main h2, .main h3 {
    color: #F2F7EC;
}
[data-testid="stWidgetLabel"] p { color: #D5E5C8 !important; }
div[role="radiogroup"] label p { color: #F2F7EC !important; }

.main-title {
    color: #B5FF35 !important;
    font-size: 42px;
    font-weight: 900;
    text-shadow: 0 0 15px rgba(168,255,0,0.4);
}
.subtitle { color: #C0D5A8 !important; font-size: 17px; }

.module-card {
    background: linear-gradient(145deg, #1D2C19, #0D130B);
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #72B82B;
    box-shadow: 0 5px 20px rgba(0,0,0,0.35);
    transition: 0.3s ease;
    margin-bottom: 14px;
}
.module-card:hover {
    border-color: #B5FF35;
    box-shadow: 0 0 18px rgba(168,255,0,0.25);
    transform: translateY(-3px);
}
.module-card h2 { color: #B5FF35 !important; }
.module-card p { color: #D5E5C8 !important; }

.stButton > button {
    background: linear-gradient(90deg, #79C900, #B5FF35);
    color: #10180A;
    border: none;
    border-radius: 12px;
    font-weight: 800;
    padding: 10px 22px;
    transition: 0.3s;
}
.stButton > button:hover {
    background: #C5FF65;
    box-shadow: 0 0 18px rgba(168,255,0,0.55);
    transform: scale(1.02);
}

.stTextInput input {
    background-color: #172216;
    color: white;
    caret-color: #B5FF35;
    border: 1px solid #72B82B;
    border-radius: 10px;
}
.stSelectbox div[data-baseweb="select"] { background-color: #172216; }
.stSelectbox div[data-baseweb="select"] * { color: #F2F7EC; }

hr { border-color: #547A32; }
.stAlert { border-radius: 12px; }
[data-testid="stSidebar"] [data-testid="stImage"] { margin-left: -8px; }

.footer-text { color: #B5FF35; text-align: center; }
</style>
"""


def aplicar_estilos():
    st.markdown(ESTILOS, unsafe_allow_html=True)


# ==========================================
# 2. VALIDACION Y LECTURA SEGURA DE TEXTO
# ==========================================

# Solo se permiten dígitos, la letra x y símbolos matemáticos básicos.
# Esto impide escribir código (no hay letras, guiones bajos ni comillas).
PATRON_VALIDO = re.compile(r"^[0-9x\s+\-*/^().=]+$")
MAX_CARACTERES = 100
MAX_EXPONENTE = 10
MAX_ANIDAMIENTO = 3

TRANSFORMACIONES = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def validar_texto(texto, permitir_igual=False):
    """Valida y normaliza el texto escrito por el usuario."""
    texto = texto.strip().lower().replace(",", ".")

    if not texto:
        raise ValueError("Escribe una expresión.")
    if len(texto) > MAX_CARACTERES:
        raise ValueError(f"Máximo {MAX_CARACTERES} caracteres.")
    if not PATRON_VALIDO.match(texto):
        raise ValueError(
            "Solo se permiten números, la letra x y los símbolos "
            "+ - * / ^ ( ) . ="
        )
    if not permitir_igual and "=" in texto:
        raise ValueError("Aquí no se usa el signo =.")
    if re.search(r"\d{7,}", texto):
        raise ValueError("Los números son demasiado grandes.")

    # Exponentes: solo enteros pequeños y sin encadenar (x^2^3)
    if re.search(r"(?:\*\*|\^)(?!\s*\d)", texto):
        raise ValueError("Los exponentes deben ser números enteros, por ejemplo x^2.")
    if re.search(r"(?:\*\*|\^)\s*\d+\s*(?:\*\*|\^)", texto):
        raise ValueError("No se permiten potencias encadenadas.")
    for e in re.findall(r"(?:\*\*|\^)\s*(\d+)", texto):
        if int(e) > MAX_EXPONENTE:
            raise ValueError(f"El exponente máximo permitido es {MAX_EXPONENTE}.")

    # Anidamiento de paréntesis
    nivel = maximo = 0
    for c in texto:
        if c == "(":
            nivel += 1
            maximo = max(maximo, nivel)
        elif c == ")":
            nivel -= 1
    if maximo > MAX_ANIDAMIENTO:
        raise ValueError("Demasiados paréntesis anidados.")

    return texto


def parsear(texto):
    """Convierte texto ya validado en una expresión de SymPy."""
    expr = parse_expr(texto, local_dict={"x": x}, transformations=TRANSFORMACIONES)
    return sp.nsimplify(expr, rational=True)  # 0.5 -> 1/2


def leer_expresion(texto):
    return parsear(validar_texto(texto, permitir_igual=False))


def leer_ecuacion(texto):
    """Devuelve (lado izquierdo, lado derecho) de una ecuación."""
    texto = validar_texto(texto, permitir_igual=True)
    partes = texto.split("=")
    if len(partes) != 2 or not partes[0].strip() or not partes[1].strip():
        raise ValueError("Escribe una ecuación con un solo signo =.")
    return parsear(partes[0]), parsear(partes[1])


def mostrar_error(e, generico="No se pudo interpretar lo escrito. Revisa la sintaxis."):
    """Muestra mensajes propios y oculta errores técnicos internos."""
    st.error(str(e) if isinstance(e, ValueError) else generico)


# ==========================================
# 3. FUNCIONES DE CONJUNTOS
# ==========================================


def ordenar(conjunto):
    """Ordena números de forma numérica y luego los textos."""
    return sorted(
        conjunto,
        key=lambda e: (1, 0, str(e)) if isinstance(e, str) else (0, e, ""),
    )


def leer_conjunto(texto):
    if len(texto) > 500:
        raise ValueError("El texto es demasiado largo.")

    try:
        datos = ast.literal_eval(texto)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        raise ValueError(
            "No se pudo leer el conjunto. Ejemplo válido: {1, 2, 3} o {'a', 'b'}."
        )

    if not isinstance(datos, (list, tuple, set)):
        raise ValueError("Debes introducir una lista o conjunto, por ejemplo {1, 2, 3}.")

    elementos = list(datos)

    if len(elementos) > 100:
        raise ValueError("Introduce como máximo 100 elementos.")
    if not all(
        isinstance(e, (int, float, str)) and not isinstance(e, bool)
        for e in elementos
    ):
        raise ValueError("Utiliza números o textos sencillos.")

    return set(elementos)


OPERACIONES = {
    "Unión": (lambda A, B: A | B, r"A \cup B", "union",
              "Elementos que están en A, en B o en ambos."),
    "Intersección": (lambda A, B: A & B, r"A \cap B", "interseccion",
                     "Elementos que están en A y también en B."),
    "Diferencia A - B": (lambda A, B: A - B, r"A - B", "dif_ab",
                         "Elementos de A que no están en B."),
    "Diferencia B - A": (lambda A, B: B - A, r"B - A", "dif_ba",
                         "Elementos de B que no están en A."),
    "Diferencia simétrica": (lambda A, B: A ^ B, r"A \triangle B", "dif_sim",
                             "Elementos que están en A o en B, pero no en ambos."),
}


def texto_region(elementos):
    if not elementos:
        return "∅"
    if len(elementos) > 6:
        return f"{len(elementos)}\nelementos"
    return "\n".join(map(str, ordenar(elementos)))


def mostrar_venn(A, B, operacion="union"):
    """Diagrama de Venn con la región de la operación resaltada."""
    centro_a, centro_b, radio = (0.40, 0.5), (0.60, 0.5), 0.27

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)

    def circulo(centro, **kw):
        return Circle(centro, radio, **kw)

    def relleno(centro, color, alpha, zorder, recorte=None):
        p = circulo(centro, facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder)
        ax.add_patch(p)
        if recorte is not None:
            p.set_clip_path(recorte)
        return p

    # Base tenue de ambos círculos
    base_a = relleno(centro_a, VERDE, 0.18, 1)
    base_b = relleno(centro_b, VERDE, 0.18, 1)

    # Región resaltada según la operación
    if operacion == "union":
        relleno(centro_a, LIMA, 0.6, 2)
        relleno(centro_b, LIMA, 0.6, 2)
    elif operacion == "interseccion":
        relleno(centro_b, LIMA, 0.75, 2, recorte=base_a)
    elif operacion == "dif_ab":
        relleno(centro_a, LIMA, 0.6, 2)
        relleno(centro_b, FONDO, 1, 3)
    elif operacion == "dif_ba":
        relleno(centro_b, LIMA, 0.6, 2)
        relleno(centro_a, FONDO, 1, 3)
    elif operacion == "dif_sim":
        relleno(centro_a, LIMA, 0.6, 2)
        relleno(centro_b, LIMA, 0.6, 2)
        relleno(centro_b, FONDO, 1, 3, recorte=base_a)

    # Contornos
    for centro in (centro_a, centro_b):
        ax.add_patch(circulo(centro, fill=False, edgecolor=LIMA, linewidth=2, zorder=5))

    # Etiquetas y elementos
    ax.text(0.22, 0.86, "CONJUNTO A", ha="center", fontsize=12,
            weight="bold", color=LIMA)
    ax.text(0.78, 0.86, "CONJUNTO B", ha="center", fontsize=12,
            weight="bold", color=LIMA)

    regiones = [
        (0.27, A - B, "normal"),
        (0.50, A & B, "bold"),
        (0.73, B - A, "normal"),
    ]
    for pos_x, elementos, peso in regiones:
        ax.text(pos_x, 0.50, texto_region(elementos), ha="center", va="center",
                fontsize=11, weight=peso, color=TEXTO, zorder=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# 4. FUNCIONES DE GRAFICOS Y FACTORIZACION
# ==========================================


def graficar_parabola(a, b, c, raices_reales):
    a, b, c = float(a), float(b), float(c)
    xv = -b / (2 * a)
    yv = c - b**2 / (4 * a)

    distancias = [abs(float(r) - xv) for r in raices_reales]
    radio = max(distancias + [3]) + 1
    xs = np.linspace(xv - radio, xv + radio, 400)
    ys = a * xs**2 + b * xs + c

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)

    ax.plot(xs, ys, color=LIMA, linewidth=2.5)
    ax.axhline(0, color=VERDE, linewidth=1)
    ax.axvline(0, color=VERDE, linewidth=1)
    ax.axvline(xv, color=VERDE, linewidth=1, linestyle="--", alpha=0.6)

    ax.scatter([xv], [yv], color="#FFD166", zorder=5, label="Vértice")
    if raices_reales:
        ax.scatter([float(r) for r in raices_reales], [0] * len(raices_reales),
                   color="#FF6B6B", zorder=5, label="Raíces")

    ax.grid(alpha=0.15, color=TEXTO)
    ax.tick_params(colors=TEXTO)
    for borde in ax.spines.values():
        borde.set_color(VERDE)
    ax.set_xlabel("x", color=TEXTO)
    ax.set_ylabel("y", color=TEXTO)
    ax.legend(facecolor=FONDO, edgecolor=VERDE, labelcolor=TEXTO)

    st.pyplot(fig)
    plt.close(fig)


def metodos_factorizacion(expresion, resultado):
    """Detecta de forma sencilla qué método se usó."""
    metodos = []
    expandida = sp.expand(expresion)
    n_terminos = len(sp.Add.make_args(expandida))

    if sp.factor_terms(expandida) != expandida:
        metodos.append("Factor común")

    factores = sp.Mul.make_args(resultado)
    binomios = [f for f in factores if isinstance(f, sp.Add) and len(f.args) == 2]
    if n_terminos == 2 and len(binomios) == 2:
        metodos.append("Diferencia de cuadrados")
    elif n_terminos == 3 and sp.Poly(expandida, x).degree() == 2 and len(binomios) >= 2:
        metodos.append("Trinomio de la forma ax² + bx + c")

    return metodos


# ==========================================
# 5. PAGINAS
# ==========================================


def pagina_inicio():
    st.markdown('<p class="main-title">📐 ALGEBRA MASTER</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Plataforma interactiva para aprender Matemática</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.header("¡Bienvenido al mundo de las Matemáticas!")
    st.write(
        "Aprende, practica y resuelve ejercicios matemáticos "
        "utilizando herramientas tecnológicas e interactivas."
    )

    tarjetas = [
        ("📘 Álgebra", "Ecuaciones, simplificación y factorización."),
        ("🧮 Resolución", "Procedimientos matemáticos paso a paso."),
        ("🧩 Conjuntos", "Unión, intersección y diferencias de conjuntos."),
        ("📊 Visualización", "Representaciones gráficas para aprender mejor."),
    ]
    col1, col2 = st.columns(2)
    for i, (titulo, texto) in enumerate(tarjetas):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(
                f'<div class="module-card"><h2>{titulo}</h2><p>{texto}</p></div>',
                unsafe_allow_html=True,
            )

    st.info("Selecciona un contenido en el menú lateral para comenzar a trabajar.")


def pagina_primer_grado():
    st.title("📘 Ecuaciones de primer grado")
    st.write("Introduce una ecuación para resolverla paso a paso.")
    st.caption("Puedes escribir 2x + 6 = 18, usar ^ para potencias y / para fracciones.")

    texto = st.text_input("Escribe tu ecuación:", value="2*x + 6 = 18",
                          max_chars=MAX_CARACTERES)

    if not st.button("Resolver ecuación"):
        return

    try:
        izq, der = leer_ecuacion(texto)
        expresion = sp.expand(izq - der)

        if expresion == 0:
            st.info("Ambos lados son iguales: la ecuación tiene **infinitas soluciones**.")
            return

        polinomio = sp.Poly(expresion, x)
        grado = polinomio.degree()

        if grado == 0:
            st.warning(
                "Al simplificar, la x desaparece y queda una igualdad falsa: "
                "la ecuación **no tiene solución**."
            )
            return
        if grado != 1:
            st.warning("Introduce una ecuación de primer grado (el mayor exponente de x debe ser 1).")
            return

        a = polinomio.coeff_monomial(x)
        b = polinomio.coeff_monomial(1)
        resultado = sp.simplify(-b / a)

        st.subheader("Procedimiento")

        st.write("**Paso 1. Ecuación original:**")
        st.latex(sp.latex(izq) + "=" + sp.latex(der))

        st.write("**Paso 2. Pasamos todo a un lado e igualamos a cero:**")
        st.latex(sp.latex(expresion) + "=0")

        st.write("**Paso 3. Despejamos el término con x:**")
        st.latex(sp.latex(a * x) + "=" + sp.latex(-b))

        st.write("**Paso 4. Dividimos entre el coeficiente de x:**")
        st.latex(f"x=\\frac{{{sp.latex(-b)}}}{{{sp.latex(a)}}}=" + sp.latex(resultado))

        st.write("**Paso 5. Verificamos reemplazando en la ecuación original:**")
        izq_v = sp.simplify(izq.subs(x, resultado))
        der_v = sp.simplify(der.subs(x, resultado))
        st.latex(f"{sp.latex(izq_v)}={sp.latex(der_v)}")

        if izq_v == der_v:
            st.success(f"Resultado: x = {resultado}  ✓ La igualdad se cumple.")
        else:
            st.error("La verificación no coincide. Revisa la ecuación.")

    except Exception as e:
        mostrar_error(e)


def pagina_cuadraticas():
    st.title("📗 Ecuaciones cuadráticas")
    st.caption("Puedes escribir x^2 - 5x + 6 = 0 o x**2 - 5*x + 6 = 0.")

    texto = st.text_input("Introduce la ecuación:", value="x**2 - 5*x + 6 = 0",
                          max_chars=MAX_CARACTERES)

    if not st.button("Resolver ecuación cuadrática"):
        return

    try:
        izq, der = leer_ecuacion(texto)
        expresion = sp.expand(izq - der)
        polinomio = sp.Poly(expresion, x) if expresion != 0 else None

        if polinomio is None or polinomio.degree() != 2:
            st.warning("Introduce una ecuación de segundo grado (el mayor exponente de x debe ser 2).")
            return

        a = polinomio.coeff_monomial(x**2)
        b = polinomio.coeff_monomial(x)
        c = polinomio.coeff_monomial(1)
        delta = sp.simplify(b**2 - 4 * a * c)

        la, lb, lc, ld = (sp.latex(v) for v in (a, b, c, delta))

        st.subheader("Procedimiento paso a paso")

        st.write("**Paso 1. Ecuación en forma general** $ax^2+bx+c=0$**:**")
        st.latex(sp.latex(expresion) + "=0")

        st.write("**Paso 2. Identificamos los coeficientes:**")
        st.latex(f"a={la},\\quad b={lb},\\quad c={lc}")

        st.write("**Paso 3. Calculamos el discriminante:**")
        st.latex(f"\\Delta=b^2-4ac=({lb})^2-4({la})({lc})={ld}")

        if delta > 0:
            st.info("Δ > 0: hay dos soluciones reales distintas.")
        elif delta == 0:
            st.info("Δ = 0: hay una solución real doble.")
        else:
            st.info("Δ < 0: no hay soluciones reales (hay dos soluciones complejas).")

        st.write("**Paso 4. Aplicamos la fórmula general:**")
        st.latex("x=\\frac{-b\\pm\\sqrt{\\Delta}}{2a}")
        st.latex(f"x=\\frac{{-({lb})\\pm\\sqrt{{{ld}}}}}{{2({la})}}")

        st.write("**Paso 5. Soluciones:**")
        soluciones = sp.solve(expresion, x)
        for i, solucion in enumerate(soluciones, 1):
            st.latex(f"x_{{{i}}}=" + sp.latex(solucion))

        st.success("Ecuación resuelta correctamente.")

        # Vértice y gráfica
        xv = sp.simplify(-b / (2 * a))
        yv = sp.simplify(c - b**2 / (4 * a))

        st.subheader("Gráfica de la parábola")
        st.latex(f"\\text{{Vértice}}=\\left({sp.latex(xv)},\\,{sp.latex(yv)}\\right)")
        raices_reales = [r for r in soluciones if sp.im(r) == 0]
        graficar_parabola(a, b, c, raices_reales)

    except Exception as e:
        mostrar_error(e)


def pagina_simplificacion():
    st.title("🧮 Simplificación algebraica")
    st.caption("Ejemplo: 3x + 2x - 4 + 7 o (x+1)^2 - x^2.")

    texto = st.text_input("Introduce una expresión:", value="3*x + 2*x - 4 + 7",
                          max_chars=MAX_CARACTERES)

    if not st.button("Simplificar"):
        return

    try:
        expresion = leer_expresion(texto)
        resultado = sp.simplify(expresion)

        st.write("**Expresión original:**")
        st.latex(sp.latex(expresion))

        st.write("**Expresión simplificada:**")
        st.latex(sp.latex(resultado))

        st.success("Simplificación completada.")

    except Exception as e:
        mostrar_error(e, "Revisa la expresión introducida.")


def pagina_factorizacion():
    st.title("📐 Factorización algebraica")
    st.caption("Ejemplos: x^2 - 9, 2x^2 - 8, x^2 + 5x + 6.")

    texto = st.text_input("Introduce un polinomio:", value="x**2 - 9",
                          max_chars=MAX_CARACTERES)

    if not st.button("Factorizar expresión"):
        return

    try:
        expresion = leer_expresion(texto)
        resultado = sp.factor(expresion)

        st.write("**Polinomio original:**")
        st.latex(sp.latex(expresion))

        st.write("**Resultado factorizado:**")
        st.latex(sp.latex(resultado))

        if resultado == sp.expand(expresion):
            st.info("Esta expresión no se puede factorizar con números enteros (es irreducible).")
        else:
            metodos = metodos_factorizacion(expresion, resultado)
            if metodos:
                st.write("**Método(s) identificado(s):** " + " + ".join(metodos))
            st.success("Factorización completada.")

    except Exception as e:
        mostrar_error(e, "No se pudo factorizar la expresión.")


def pagina_conjuntos():
    st.title("🧩 Operaciones con conjuntos")
    st.write("Introduce los elementos de los conjuntos utilizando llaves.")

    col1, col2 = st.columns(2)
    with col1:
        entrada_a = st.text_input("Conjunto A:", value="{1, 2, 3, 4, 5}")
    with col2:
        entrada_b = st.text_input("Conjunto B:", value="{3, 4, 5, 6, 7}")

    operacion = st.selectbox("Selecciona una operación:", list(OPERACIONES))

    if not st.button("Resolver operación"):
        return

    try:
        A = leer_conjunto(entrada_a)
        B = leer_conjunto(entrada_b)
        funcion, simbolo, clave, definicion = OPERACIONES[operacion]
        resultado = funcion(A, B)

        st.subheader("Conjuntos ingresados")
        st.write("A =", ordenar(A))
        st.write("B =", ordenar(B))

        st.subheader("Procedimiento")
        st.write(f"**Operación:** {operacion}")
        st.caption(definicion)
        st.latex(simbolo)

        st.write("**Resultado:**")
        st.code("{" + ", ".join(map(str, ordenar(resultado))) + "}", language=None)

        st.success(f"Cantidad de elementos del resultado: {len(resultado)}")

        st.subheader("Diagrama de Venn")
        st.caption("La región resaltada corresponde al resultado de la operación.")
        mostrar_venn(A, B, clave)

    except Exception as e:
        mostrar_error(e, "Error al leer los conjuntos.")


# ==========================================
# 6. MENU LATERAL Y NAVEGACION
# ==========================================

PAGINAS = {
    "Inicio": pagina_inicio,
    "Ecuaciones de primer grado": pagina_primer_grado,
    "Ecuaciones cuadráticas": pagina_cuadraticas,
    "Simplificación algebraica": pagina_simplificacion,
    "Factorización": pagina_factorizacion,
    "Operaciones con conjuntos": pagina_conjuntos,
}


def main():
    aplicar_estilos()

    with st.sidebar:
        try:
            st.image("logo_charagua.png", width=155)
        except Exception:
            st.warning("No se encontró logo_charagua.png")

        st.markdown("## ALGEBRA MASTER")
        st.caption("UNIDAD ACADÉMICA CHARAGUA")
        st.divider()

        opcion = st.radio("MENÚ PRINCIPAL", list(PAGINAS))

        st.divider()
        st.caption("Recurso educativo de Matemática")
        st.caption("Formación Docente")

    PAGINAS[opcion]()

    st.divider()
    st.markdown(
        """
        <div class="footer-text">
        <b>ALGEBRA MASTER</b><br>
        Unidad Académica Charagua<br>
        <b>Equipo de trabajo: Ariel - Eliana - Pamela - Mayte - Karen - Tatiana</b><br>
        <b> Mat "B"</b><br>
        <b>Docente: Victor Anibal</b><br>
        Plataforma educativa de Matemática
        </div>
        """,
        unsafe_allow_html=True,
    )


main()
