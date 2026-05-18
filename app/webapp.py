import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuração da página
st.set_page_config(
    page_title="Previsão de Lesões - Football Injury Predictor",
    page_icon="⚽",
    layout="wide"
)

# Título
st.title("Sistema de Previsão de Risco de Lesões")
st.markdown("### Projeto IART - Machine Learning para Prevenção de Lesões em Futebol")

st.markdown("---")

# sidebar - info do projeto
with st.sidebar:
    st.header("Sobre o Projeto")
    st.markdown("""
    **Cliente:** FC Porto
    
    **Problema:** Prevenir lesões em jogadores de futebol através da análise de dados físicos e de treino.
    
    **Solução:** Sistema de ML que classifica jogadores em 3 níveis de risco:
    - 🟢 **Baixo Risco** (0)
    - 🟡 **Risco Médio** (1)  
    - 🔴 **Alto Risco** (2)
    
    **Modelos Testados:**
    - Logistic Regression
    - Decision Tree
    - Random Forest
    """)
    
    st.markdown("---")
    st.markdown("**Desenvolvido por:**")
    st.markdown("LEIC IART - Grupo 24:")
    st.markdown("João Moreira")
    st.markdown("Martim Leme")
    st.markdown("Dinis Bahia")


# Verificar se modelo existe
model_path = "models/injury_model.pkl"

if not os.path.exists(model_path):
    st.error("Modelo não encontrado! Por favor treina primeiro o modelo executando `randomforest.py`")
    st.stop()

# Carregar modelo
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

st.success("Modelo Random Forest carregado com sucesso!")

# Criar tabs
tab1, tab2, tab3 = st.tabs(["Previsão Individual", "Previsão em Lote", "Sobre o Modelo"])

# TAB 1: Previsão Individual
with tab1:
    st.header("Insira os dados do jogador")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Dados Pessoais")
        nome = st.text_input("Nome do Jogador", "Jogador Exemplo")
        idade = st.number_input("Idade", min_value=16, max_value=45, value=25)
        
        st.subheader("Capacidades Físicas")
        altura_salto = st.slider("Altura de Salto (cm)", 25.0, 60.0, 42.5, 0.1)
        capacidade_aerobica = st.slider("Capacidade Aeróbica", 45.0, 100.0, 72.5, 0.1)
        capacidade_anaerobica = st.slider("Capacidade Anaeróbica", 45.0, 100.0, 72.5, 0.1)
    
    with col2:
        st.subheader("Dados de Jogo")
        distancia_jogo = st.number_input("Distância Percorrida em Jogo (km)", 
                                         min_value=0.0, max_value=15.0, value=8.5, step=0.1)
        sprints_jogo = st.number_input("Sprints em Jogo", 
                                       min_value=0, max_value=100, value=35)
        
        st.subheader("Dados de Treino")
        distancia_treino = st.number_input("Distância Percorrida em Treino (km)", 
                                           min_value=0.0, max_value=50.0, value=28.0, step=0.1)
        sprints_treino = st.number_input("Sprints em Treino", 
                                         min_value=0, max_value=200, value=85)
    
    with col3:
        st.subheader("Histórico Médico")
        lesao_recente = st.selectbox("Lesão Recente?", 
                                     options=[0, 1], 
                                     format_func=lambda x: "Não" if x == 0 else "Sim")
        ausencia_dores = st.selectbox("Ausência de Dores?", 
                                      options=[0, 1],
                                      format_func=lambda x: "Tem dores" if x == 0 else "Sem dores")
        
        st.subheader("Gestão de Carga")
        tempo_descanso = st.slider("Tempo de Descanso entre Jogos (dias)", 
                                   2, 9, 5)
    
    st.markdown("---")
    
    # Botão de previsão
    if st.button("Fazer Previsão", type="primary", use_container_width=True):
        
        # Preparar dados
        input_data = pd.DataFrame({
            'idade': [idade],
            'distancia_jogo': [distancia_jogo],
            'distancia_treino': [distancia_treino],
            'sprints_jogo': [sprints_jogo],
            'sprints_treino': [sprints_treino],
            'altura_salto': [altura_salto],
            'capacidade_aerobica': [capacidade_aerobica],
            'capacidade_anaerobica': [capacidade_anaerobica],
            'lesao_recente': [lesao_recente],
            'ausencia_dores': [ausencia_dores],
            'tempo_descanso_entre_jogos': [tempo_descanso]
        })
        
        # Fazer previsão
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        
        # Mostrar resultado
        st.markdown("### Resultado da Previsão")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            if prediction == 0:
                st.success(f"### 🟢 BAIXO RISCO")
                st.metric("Jogador", nome)
                st.metric("Classe Prevista", "0 - Baixo Risco")
            elif prediction == 1:
                st.warning(f"### 🟡 RISCO MÉDIO")
                st.metric("Jogador", nome)
                st.metric("Classe Prevista", "1 - Risco Médio")
            else:
                st.error(f"### 🔴 ALTO RISCO")
                st.metric("Jogador", nome)
                st.metric("Classe Prevista", "2 - Alto Risco")
        
        with col_res2:
            st.metric("Probabilidade Baixo Risco", f"{proba[0]*100:.1f}%")
            st.metric("Probabilidade Risco Médio", f"{proba[1]*100:.1f}%")
            st.metric("Probabilidade Alto Risco", f"{proba[2]*100:.1f}%")
        
        with col_res3:
            st.markdown("#### Recomendações")
            if prediction == 0:
                st.info("""
                Jogador pode continuar treino normal
                
                Manter monitorização regular
                """)
            elif prediction == 1:
                st.warning("""
                Reduzir intensidade dos treinos
                
                Aumentar tempo de recuperação
                
                Avaliação médica recomendada
                """)
            else:
                st.error("""
                Parar treinos intensos imediatamente
                
                Avaliação médica obrigatória
                
                Período de descanso necessário
                """)
        
        # Gráfico de probabilidades
        st.markdown("---")
        st.markdown("#### Distribuição de Probabilidades")
        
        prob_df = pd.DataFrame({
            'Risco': ['Baixo (0)', 'Médio (1)', 'Alto (2)'],
            'Probabilidade': proba * 100
        })
        
        st.bar_chart(prob_df.set_index('Risco'))

# TAB 2: Previsão em Lote
with tab2:
    st.header("Upload de Dataset para Previsão em Lote")
    
    st.markdown("""
    Faça upload de um ficheiro CSV com as seguintes colunas:
    - `idade`, `distancia_jogo`, `distancia_treino`, `sprints_jogo`, `sprints_treino`
    - `altura_salto`, `capacidade_aerobica`, `capacidade_anaerobica`
    - `lesao_recente`, `ausencia_dores`, `tempo_descanso_entre_jogos`
    
    Opcionalmente pode incluir a coluna `nome` para identificação dos jogadores.
    """)
    
    uploaded_file = st.file_uploader("Escolha um ficheiro CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Ler CSV
            df_batch = pd.read_csv(uploaded_file)
            
            st.success(f"Ficheiro carregado: {len(df_batch)} jogadores")
            
            # Mostrar preview
            st.subheader("Preview dos Dados")
            st.dataframe(df_batch.head())
            
            # Verificar se tem coluna nome
            has_name = 'nome' in df_batch.columns
            
            # Preparar dados para previsão
            if has_name:
                X_batch = df_batch.drop(columns=['nome'])
                names = df_batch['nome']
            else:
                X_batch = df_batch.copy()
                names = [f"Jogador_{i+1}" for i in range(len(df_batch))]
            
            # Se tiver risco_lesao (para comparação), remover
            if 'risco_lesao' in X_batch.columns:
                y_true = X_batch['risco_lesao']
                X_batch = X_batch.drop(columns=['risco_lesao'])
                has_true_labels = True
            else:
                has_true_labels = False
            
            # Fazer previsões
            if st.button("🔮 Fazer Previsões em Lote", type="primary"):
                predictions = model.predict(X_batch)
                probas = model.predict_proba(X_batch)
                
                # Criar DataFrame de resultados
                results_df = pd.DataFrame({
                    'Nome': names,
                    'Risco Previsto': predictions,
                    'Prob. Baixo (%)': probas[:, 0] * 100,
                    'Prob. Médio (%)': probas[:, 1] * 100,
                    'Prob. Alto (%)': probas[:, 2] * 100
                })
                
                if has_true_labels:
                    results_df['Risco Real'] = y_true.values
                    # Corrigir: converter para Series primeiro antes de usar .map()
                    correct_array = (predictions == y_true.values)
                    results_df['Correto?'] = pd.Series(correct_array).map({True: '✅', False: '❌'}).values

                
                st.subheader("📋 Resultados das Previsões")
                st.dataframe(results_df)
                
                # Estatísticas
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    baixo = (predictions == 0).sum()
                    st.metric("🟢 Baixo Risco", baixo, 
                             f"{baixo/len(predictions)*100:.1f}%")
                
                with col_stat2:
                    medio = (predictions == 1).sum()
                    st.metric("🟡 Risco Médio", medio,
                             f"{medio/len(predictions)*100:.1f}%")
                
                with col_stat3:
                    alto = (predictions == 2).sum()
                    st.metric("🔴 Alto Risco", alto,
                             f"{alto/len(predictions)*100:.1f}%")
                
                if has_true_labels:
                    from sklearn.metrics import accuracy_score
                    acc = accuracy_score(y_true, predictions)
                    st.success(f"Precisão nas previsões: {acc*100:.1f}%")
                
                # Download dos resultados
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Resultados (CSV)",
                    data=csv,
                    file_name='previsoes_lesoes.csv',
                    mime='text/csv'
                )
        
        except Exception as e:
            st.error(f"Erro ao processar ficheiro: {e}")

# TAB 3: Sobre o Modelo
with tab3:
    st.header("Informações sobre o Modelo")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.subheader("Modelo Utilizado")
        st.markdown("""
        **Random Forest Classifier**
        
        - **N° de Árvores:** 100
        - **Profundidade Máxima:** 15
        - **Random State:** 42
        
        **Porquê Random Forest?**
        - Robusto contra overfitting
        - Boa performance com dados tabulares
        - Fornece importância das features
        - Lida bem com relações não-lineares
        """)
        
        st.subheader("Dataset de Treino")
        st.markdown("""
        - **Total de amostras:** 500 jogadores
        - **Features:** 11 variáveis
        - **Classes:** 3 níveis de risco
        - **Split:** 80% treino, 20% teste
        """)
    
    with col_info2:
        st.subheader("Performance do Modelo")
        
        # Tentar carregar resultados se existirem
        if os.path.exists("results/rf_results.json"):
            import json
            with open("results/rf_results.json", 'r') as f:
                results = json.load(f)
            
            st.metric("Accuracy", f"{results['accuracy']:.3f}")
            st.metric("F1-Score", f"{results['f1_score']:.3f}")
            st.metric("CV Mean", f"{results['cv_mean']:.3f}")
        else:
            st.info("Execute os modelos para ver as métricas detalhadas")
        
        st.subheader("Features Utilizadas")
        features = [
            "Idade", "Distância em Jogo", "Distância em Treino",
            "Sprints em Jogo", "Sprints em Treino", "Altura de Salto",
            "Capacidade Aeróbica", "Capacidade Anaeróbica",
            "Lesão Recente", "Ausência de Dores", "Tempo de Descanso"
        ]
        for feat in features:
            st.markdown(f"- {feat}")
    
    st.markdown("---")
    
    st.subheader("Limitações e Considerações")
    st.warning("""
    - Este é um **POC (Proof of Concept)** com dados artificiais
    - Em produção, seria necessário dados reais validados
    - Recomenda-se validação médica profissional
    - O modelo deve ser retreinado periodicamente com novos dados
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>⚽ Football Injury Predictor | IART Project | LEIC 2025/26</p>
</div>
""", unsafe_allow_html=True)