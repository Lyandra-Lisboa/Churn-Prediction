import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

def preprocess_ligga_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pré-processa dados da Ligga para análise de clusterização
    Adapta os dados reais para o formato esperado pelo modelo K-Modes
    """
    logger.info("🔧 Iniciando pré-processamento dos dados da Ligga...")
    
    # Fazer cópia para não alterar o original
    df_processed = df.copy()
    
    # 1. CALCULAR IDADE A PARTIR DA DATA DE NASCIMENTO
    df_processed = _calculate_age(df_processed)
    
    # 2. CRIAR FEATURES DE SVAs (SERVIÇOS ADICIONAIS)
    df_processed = _create_sva_features(df_processed)
    
    # 3. CRIAR FEATURE DE FIDELIZAÇÃO
    df_processed = _create_fidelizacao_feature(df_processed)
    
    # 4. CRIAR FEATURE DE MEIO DE COBRANÇA
    df_processed = _create_meio_cobranca_feature(df_processed)
    
    # 5. ESTIMAR SCORE SERASA
    df_processed = _estimate_score_serasa(df_processed)
    
    # 6. ESTIMAR RENDA MÉDIA
    df_processed = _estimate_renda_media(df_processed)
    
    # 7. FILTRAR DADOS VÁLIDOS
    df_processed = _filter_valid_data(df_processed)
    
    # 8. SELECIONAR COLUNAS PARA ANÁLISE
    df_analysis = _select_analysis_columns(df_processed)
    
    # 9. CONVERTER TIPOS DE DADOS
    df_analysis = _convert_data_types(df_analysis)
    
    # 10. VALIDAR DADOS FINAIS
    _validate_processed_data(df_analysis)
    
    logger.info(f"✅ Pré-processamento concluído: {len(df_analysis):,} registros, {len(df_analysis.columns)} colunas")
    
    return df_analysis

def _calculate_age(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula idade a partir da data de nascimento"""
    logger.info("📅 Calculando idade dos clientes...")
    
    if 'data_nascimento' in df.columns:
        # Converter para datetime
        df['data_nascimento'] = pd.to_datetime(df['data_nascimento'], errors='coerce')
        
        # Calcular idade
        hoje = pd.Timestamp.now()
        df['idade'] = (hoje - df['data_nascimento']).dt.days / 365.25
        df['idade'] = df['idade'].round(0).astype('Int64')
        
        # Para valores inválidos, usar uma idade padrão baseada no perfil
        df['idade'] = df['idade'].fillna(35)  # Idade padrão
        
        logger.info(f"   • Idade média: {df['idade'].mean():.1f} anos")
        logger.info(f"   • Faixa etária: {df['idade'].min():.0f} - {df['idade'].max():.0f} anos")
    else:
        # Se não temos data de nascimento, criar idade padrão baseada no valor do contrato
        logger.warning("⚠️  Campo 'data_nascimento' não encontrado, criando idade estimada...")
        df['idade'] = 35  # Idade padrão
        
        # Ajustar idade baseada no valor do contrato (aproximação)
        if 'valor_contrato' in df.columns:
            # Clientes com valor maior tendem a ser mais velhos
            mask_alto_valor = df['valor_contrato'] > 150
            mask_baixo_valor = df['valor_contrato'] < 100
            
            df.loc[mask_alto_valor, 'idade'] = 45
            df.loc[mask_baixo_valor, 'idade'] = 28
    
    return df

def _create_sva_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de SVAs (Serviços de Valor Agregado)"""
    logger.info("📺 Criando features de SVAs...")
    
    # Mapear produtos para SVAs baseado na análise da Ligga
    # Mesh (WiFi)
    df['mesh_ponto_1'] = 'Não'
    df['mesh_ponto_2'] = 'Não' 
    df['mesh_ponto_3'] = 'Não'
    
    if 'produto_final' in df.columns:
        # Identificar produtos com Mesh
        mesh_keywords = ['mesh', 'wifi', 'wi-fi', 'wireless']
        produto_lower = df['produto_final'].astype(str).str.lower()
        
        mesh_mask = produto_lower.apply(lambda x: any(keyword in x for keyword in mesh_keywords))
        df.loc[mesh_mask, 'mesh_ponto_1'] = 'Sim'
    
    # Clube Ligga (assumir que clientes com maior valor têm mais benefícios)
    df['clube_ligga'] = 'Não'
    if 'valor_contrato' in df.columns:
        # Clientes com valor acima de R$120 têm clube
        df.loc[df['valor_contrato'] > 120, 'clube_ligga'] = 'Sim'
    
    # HBO Max e Paramount+ (baseado no valor e tipo de produto)
    df['hbo_max'] = 'Não'
    df['paramount_plus'] = 'Não'
    
    if 'valor_contrato' in df.columns and 'produto_final' in df.columns:
        # Clientes premium têm streaming
        produto_lower = df['produto_final'].astype(str).str.lower()
        valor_alto = df['valor_contrato'] > 140
        
        # HBO para clientes premium
        hbo_keywords = ['hbo', 'premium', 'max']
        hbo_mask = produto_lower.apply(lambda x: any(keyword in x for keyword in hbo_keywords))
        df.loc[valor_alto | hbo_mask, 'hbo_max'] = 'Sim'
        
        # Paramount+ para clientes de valor médio-alto
        paramount_keywords = ['paramount', 'plus', 'streaming']
        paramount_mask = produto_lower.apply(lambda x: any(keyword in x for keyword in paramount_keywords))
        df.loc[(df['valor_contrato'] > 110) | paramount_mask, 'paramount_plus'] = 'Sim'
    
    # Seguro Residencial (baseado no perfil do cliente)
    df['seguro_residencial'] = 'Não'
    if 'valor_contrato' in df.columns and 'aging_meses' in df.columns:
        # Clientes antigos e de valor médio-alto têm seguro
        seguro_mask = (df['valor_contrato'] > 130) & (df['aging_meses'] > 24)
        df.loc[seguro_mask, 'seguro_residencial'] = 'Sim'
    
    logger.info("   • SVAs criados: Mesh, Clube Ligga, HBO Max, Paramount+, Seguro")
    
    return df

def _create_fidelizacao_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Cria feature de fidelização"""
    logger.info("💍 Criando feature de fidelização...")
    
    df['cliente_fidelizado'] = 'Não'
    
    # Critérios para fidelização (baseado na análise da Ligga):
    # - Clientes antigos (>24 meses)
    # - Sem problemas de pagamento
    # - Poucos atendimentos negativos
    
    criterios_fidelizacao = pd.Series(True, index=df.index)
    
    # Critério 1: Aging > 24 meses
    if 'aging_meses' in df.columns:
        criterios_fidelizacao &= (df['aging_meses'] > 24)
    
    # Critério 2: Baixo risco de churn
    if 'is_churner' in df.columns:
        criterios_fidelizacao &= (df['is_churner'] == 'ativo')
    
    # Critério 3: Poucos atendimentos negativos
    if 'atendimentos_negativos' in df.columns:
        criterios_fidelizacao &= (df['atendimentos_negativos'] <= 1)
    
    # Critério 4: Valor do contrato acima da média
    if 'valor_contrato' in df.columns:
        valor_medio = df['valor_contrato'].median()
        criterios_fidelizacao &= (df['valor_contrato'] >= valor_medio)
    
    df.loc[criterios_fidelizacao, 'cliente_fidelizado'] = 'Sim'
    
    pct_fidelizados = (df['cliente_fidelizado'] == 'Sim').mean() * 100
    logger.info(f"   • {pct_fidelizados:.1f}% dos clientes identificados como fidelizados")
    
    return df

def _create_meio_cobranca_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Cria feature de meio de cobrança"""
    logger.info("💳 Criando feature de meio de cobrança...")
    
    # Padrão: Boleto (mais comum no Brasil)
    df['tipo_meio_cobranca'] = 'Boleto'
    
    # Critérios para débito automático:
    # - Clientes de maior valor
    # - Clientes fidelizados
    # - Histórico de bom pagamento
    
    if 'valor_contrato' in df.columns:
        # Clientes com valor alto tendem a usar débito automático
        valor_alto = df['valor_contrato'] > df['valor_contrato'].quantile(0.7)
        df.loc[valor_alto, 'tipo_meio_cobranca'] = 'Débito Automático'
    
    if 'cliente_fidelizado' in df.columns:
        # Clientes fidelizados preferem débito automático
        fidelizados = df['cliente_fidelizado'] == 'Sim'
        df.loc[fidelizados, 'tipo_meio_cobranca'] = 'Débito Automático'
    
    if 'atendimentos_negativos' in df.columns:
        # Clientes sem problemas usam débito automático
        sem_problemas = df['atendimentos_negativos'] == 0
        valor_medio_alto = df['valor_contrato'] > df['valor_contrato'].median()
        df.loc[sem_problemas & valor_medio_alto, 'tipo_meio_cobranca'] = 'Débito Automático'
    
    pct_debito = (df['tipo_meio_cobranca'] == 'Débito Automático').mean() * 100
    logger.info(f"   • {pct_debito:.1f}% dos clientes com débito automático")
    
    return df

def _estimate_score_serasa(df: pd.DataFrame) -> pd.DataFrame:
    """Estima score Serasa baseado no comportamento do cliente"""
    logger.info("📊 Estimando Score Serasa...")
    
    # Score base
    df['score_serasa_final'] = 650
    
    # Ajustes baseados no comportamento
    
    # Fator 1: Valor do contrato (maior valor = maior score)
    if 'valor_contrato' in df.columns:
        valor_norm = (df['valor_contrato'] - df['valor_contrato'].min()) / (df['valor_contrato'].max() - df['valor_contrato'].min())
        df['score_serasa_final'] += (valor_norm * 200).fillna(0)
    
    # Fator 2: Aging (clientes antigos = maior score)
    if 'aging_meses' in df.columns:
        aging_bonus = np.minimum(df['aging_meses'] * 2, 100)  # Máximo 100 pontos
        df['score_serasa_final'] += aging_bonus.fillna(0)
    
    # Fator 3: Atendimentos negativos (penalizar)
    if 'atendimentos_negativos' in df.columns:
        penalidade = df['atendimentos_negativos'] * 50  # -50 pontos por atendimento negativo
        df['score_serasa_final'] -= penalidade.fillna(0)
    
    # Fator 4: Status de churn (penalizar churners)
    if 'is_churner' in df.columns:
        churn_penalty = df['is_churner'].map({
            'ativo': 0,
            'solicitou_cancel': -100,
            'cancelado': -200
        }).fillna(0)
        df['score_serasa_final'] += churn_penalty
    
    # Fator 5: Fidelização (bônus)
    if 'cliente_fidelizado' in df.columns:
        fidelizacao_bonus = (df['cliente_fidelizado'] == 'Sim') * 50
        df['score_serasa_final'] += fidelizacao_bonus
    
    # Limitar score entre 300 e 950
    df['score_serasa_final'] = df['score_serasa_final'].clip(300, 950).round(0).astype(int)
    
    score_medio = df['score_serasa_final'].mean()
    logger.info(f"   • Score médio estimado: {score_medio:.0f}")
    
    return df

def _estimate_renda_media(df: pd.DataFrame) -> pd.DataFrame:
    """Estima renda média baseada na região e valor do contrato"""
    logger.info("💰 Estimando renda média...")
    
    # Renda base por região (em salários mínimos)
    df['renda_media'] = 2.5  # Padrão
    
    if 'regiao' in df.columns:
        # Curitiba tem renda média mais alta
        df.loc[df['regiao'] == 'curitiba', 'renda_media'] = 3.2
        df.loc[df['regiao'] == 'interior', 'renda_media'] = 2.1
    
    # Ajustar baseado no valor do contrato
    if 'valor_contrato' in df.columns:
        # Clientes com valor alto têm renda proporcionalmente maior
        multiplicador = 1 + (df['valor_contrato'] - 100) / 1000  # Ajuste sutil
        df['renda_media'] *= multiplicador.clip(0.5, 5.0)  # Limitar entre 0.5 e 5 SM
    
    df['renda_media'] = df['renda_media'].round(1)
    
    renda_media_geral = df['renda_media'].mean()
    logger.info(f"   • Renda média estimada: {renda_media_geral:.1f} salários mínimos")
    
    return df

def _filter_valid_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra dados válidos para análise"""
    logger.info("🔍 Filtrando dados válidos...")
    
    inicial_count = len(df)
    
    # Filtro 1: Idade válida
    if 'idade' in df.columns:
        df = df[(df['idade'] >= 15) & (df['idade'] <= 100)]
        logger.info(f"   • Filtro idade (15-100): {len(df):,} registros mantidos")
    
    # Filtro 2: Valor do contrato válido
    if 'valor_contrato' in df.columns:
        df = df[df['valor_contrato'] > 0]
        logger.info(f"   • Filtro valor > 0: {len(df):,} registros mantidos")
    
    # Filtro 3: Contrato válido
    if 'contrato' in df.columns:
        df = df[df['contrato'].notna()]
        logger.info(f"   • Filtro contrato válido: {len(df):,} registros mantidos")
    
    removidos = inicial_count - len(df)
    if removidos > 0:
        logger.info(f"   • Total removido: {removidos:,} registros ({(removidos/inicial_count)*100:.1f}%)")
    
    return df

def _select_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona colunas para análise"""
    logger.info("📋 Selecionando colunas para análise...")
    
    # Colunas equivalentes às do sistema original, adaptadas para Ligga
    col_analysis = [
        'idade',
        'regiao', 
        'tipo_meio_cobranca',
        'mesh_ponto_1',
        'mesh_ponto_2', 
        'mesh_ponto_3',
        'clube_ligga',
        'hbo_max',
        'paramount_plus',
        'seguro_residencial',
        'cliente_fidelizado',
        'velocidade_num',  # equivale a 'v_velocidade'
        'score_serasa_final',
        'renda_media',
        'total_atendimentos',  # equivale a 'qtde_massivas'
        'atendimentos_negativos',  # equivale a 'qtde_reparos'
        'valor_contrato'  # equivale a 'valor_circuito'
    ]
    
    # Verificar quais colunas existem
    colunas_existentes = [col for col in col_analysis if col in df.columns]
    colunas_faltantes = [col for col in col_analysis if col not in df.columns]
    
    if colunas_faltantes:
        logger.warning(f"   ⚠️  Colunas não encontradas: {colunas_faltantes}")
    
    # Criar DataFrame apenas com colunas existentes
    df_analysis = df[colunas_existentes].copy()
    
    logger.info(f"   • Selecionadas {len(colunas_existentes)} de {len(col_analysis)} colunas")
    
    return df_analysis

def _convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converte tipos de dados apropriados"""
    logger.info("🔄 Convertendo tipos de dados...")
    
    # Colunas categóricas
    cat_cols = [
        'regiao', 'tipo_meio_cobranca', 'mesh_ponto_1', 'mesh_ponto_2',
        'mesh_ponto_3', 'clube_ligga', 'hbo_max', 'paramount_plus',
        'seguro_residencial', 'cliente_fidelizado'
    ]
    
    # Converter apenas colunas que existem
    cat_cols_existentes = [col for col in cat_cols if col in df.columns]
    
    for col in cat_cols_existentes:
        df[col] = df[col].astype('category')
        logger.info(f"   • {col}: {df[col].nunique()} categorias")
    
    # Converter colunas numéricas
    numeric_cols = ['idade', 'velocidade_num', 'score_serasa_final', 'renda_media', 
                   'total_atendimentos', 'atendimentos_negativos', 'valor_contrato']
    
    numeric_cols_existentes = [col for col in numeric_cols if col in df.columns]
    
    for col in numeric_cols_existentes:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    logger.info(f"   • Convertidas {len(cat_cols_existentes)} categóricas e {len(numeric_cols_existentes)} numéricas")
    
    return df

def _validate_processed_data(df: pd.DataFrame):
    """Valida dados processados"""
    logger.info("✅ Validando dados processados...")
    
    # Verificar valores nulos
    null_counts = df.isnull().sum()
    colunas_com_nulos = null_counts[null_counts > 0]
    
    if len(colunas_com_nulos) > 0:
        logger.warning("   ⚠️  Colunas com valores nulos:")
        for col, count in colunas_com_nulos.items():
            pct = (count / len(df)) * 100
            logger.warning(f"     • {col}: {count:,} ({pct:.1f}%)")
    else:
        logger.info("   • Nenhum valor nulo encontrado")
    
    # Verificar tipos de dados
    logger.info("   • Tipos de dados:")
    categorical_count = len(df.select_dtypes(include=['category']).columns)
    numeric_count = len(df.select_dtypes(include=['number']).columns)
    logger.info(f"     • Categóricas: {categorical_count}")
    logger.info(f"     • Numéricas: {numeric_count}")
    
    # Estatísticas básicas
    if 'valor_contrato' in df.columns:
        logger.info(f"   • Valor médio dos contratos: R$ {df['valor_contrato'].mean():.2f}")
    
    if 'idade' in df.columns:
        logger.info(f"   • Idade média: {df['idade'].mean():.1f} anos")

def preprocess_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função de compatibilidade com o código original
    Chama o pré-processamento adaptado para a Ligga
    """
    return preprocess_ligga_data(df)

if __name__ == "__main__":
    # Teste do pré-processamento
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testando pré-processamento da Ligga...")
    
    # Criar dados de exemplo
    data_exemplo = {
        'contrato': ['12345', '67890', '11111'],
        'valor_contrato': [120.0, 85.0, 180.0],
        'aging_meses': [24, 6, 48],
        'regiao': ['curitiba', 'interior', 'curitiba'],
        'data_nascimento': ['1985-05-15', '1992-12-20', '1978-03-10'],
        'produto_final': ['Internet 300MB', 'Internet Mesh 500MB', 'Internet Premium'],
        'is_churner': ['ativo', 'ativo', 'cancelado'],
        'total_atendimentos': [2, 0, 5],
        'atendimentos_negativos': [0, 0, 3],
        'velocidade_num': [300, 500, 800]
    }
    
    df_teste = pd.DataFrame(data_exemplo)
    
    # Pré-processar
    df_processado = preprocess_ligga_data(df_teste)
    
    print(f"\n✅ Pré-processamento concluído!")
    print(f"📊 Dados processados: {len(df_processado)} registros, {len(df_processado.columns)} colunas")
    print(f"\nColunas finais:")
    for i, col in enumerate(df_processado.columns):
        print(f"  {i+1:2d}. {col} ({df_processado[col].dtype})")
