import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validador de qualidade dos dados extraídos"""
    
    def __init__(self):
        self.validation_rules = self._define_validation_rules()
    
    def _define_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define regras de validação para diferentes tipos de dados"""
        return {
            'cliente_id': {
                'required': True,
                'unique': True,
                'data_type': 'string'
            },
            'idade': {
                'required': True,
                'min_value': 18,
                'max_value': 100,
                'data_type': 'numeric'
            },
            'valor_mensalidade': {
                'required': True,
                'min_value': 0,
                'max_value': 10000,
                'data_type': 'numeric'
            },
            'tempo_base_meses': {
                'required': True,
                'min_value': 0,
                'max_value': 600,  # 50 anos máximo
                'data_type': 'numeric'
            },
            'faturas_atraso_12m': {
                'required': False,
                'min_value': 0,
                'max_value': 12,
                'data_type': 'numeric'
            }
        }
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Executa validação completa do dataset"""
        logger.info(f"🔍 Validando dataset com {len(df):,} registros...")
        
        results = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'validation_summary': {},
            'quality_score': 0,
            'critical_issues': [],
            'warnings': [],
            'column_analysis': {}
        }
        
        # Validar cada coluna
        for column in df.columns:
            if column in self.validation_rules:
                column_results = self._validate_column(df, column)
                results['column_analysis'][column] = column_results
                
                # Adicionar issues críticos
                if column_results.get('critical_issues'):
                    results['critical_issues'].extend(column_results['critical_issues'])
                
                # Adicionar warnings
                if column_results.get('warnings'):
                    results['warnings'].extend(column_results['warnings'])
        
        # Validações gerais
        general_validation = self._validate_general_quality(df)
        results.update(general_validation)
        
        # Calcular score de qualidade
        results['quality_score'] = self._calculate_quality_score(results)
        
        # Log dos resultados
        self._log_validation_results(results)
        
        return results
    
    def _validate_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Valida uma coluna específica"""
        rules = self.validation_rules[column]
        results = {
            'missing_count': 0,
            'missing_percentage': 0,
            'critical_issues': [],
            'warnings': [],
            'stats': {}
        }
        
        if column not in df.columns:
            results['critical_issues'].append(f"Coluna obrigatória '{column}' não encontrada")
            return results
        
        col_data = df[column]
        
        # Verificar valores faltantes
        missing_count = col_data.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        results['missing_count'] = int(missing_count)
        results['missing_percentage'] = round(missing_pct, 2)
        
        if rules.get('required', False) and missing_count > 0:
            if missing_pct > 10:
                results['critical_issues'].append(
                    f"Coluna obrigatória '{column}' tem {missing_pct:.1f}% de valores faltantes"
                )
            else:
                results['warnings'].append(
                    f"Coluna '{column}' tem {missing_count} valores faltantes"
                )
        
        # Verificar duplicatas (se aplicável)
        if rules.get('unique', False):
            duplicates = col_data.duplicated().sum()
            if duplicates > 0:
                results['critical_issues'].append(
                    f"Coluna '{column}' deveria ser única mas tem {duplicates} duplicatas"
                )
        
        # Verificar valores numéricos
        if rules.get('data_type') == 'numeric':
            numeric_data = pd.to_numeric(col_data, errors='coerce')
            
            # Valores que não puderam ser convertidos
            conversion_errors = numeric_data.isnull().sum() - missing_count
            if conversion_errors > 0:
                results['warnings'].append(
                    f"Coluna '{column}' tem {conversion_errors} valores não numéricos"
                )
            
            # Range de valores
            if 'min_value' in rules:
                below_min = (numeric_data < rules['min_value']).sum()
                if below_min > 0:
                    results['warnings'].append(
                        f"Coluna '{column}' tem {below_min} valores abaixo do mínimo ({rules['min_value']})"
                    )
            
            if 'max_value' in rules:
                above_max = (numeric_data > rules['max_value']).sum()
                if above_max > 0:
                    results['warnings'].append(
                        f"Coluna '{column}' tem {above_max} valores acima do máximo ({rules['max_value']})"
                    )
            
            # Estatísticas descritivas
            if not numeric_data.dropna().empty:
                results['stats'] = {
                    'mean': round(numeric_data.mean(), 2),
                    'median': round(numeric_data.median(), 2),
                    'std': round(numeric_data.std(), 2),
                    'min': round(numeric_data.min(), 2),
                    'max': round(numeric_data.max(), 2)
                }
        
        return results
    
    def _validate_general_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validações gerais de qualidade"""
        results = {
            'duplicate_rows': 0,
            'empty_rows': 0,
            'data_types_summary': {}
        }
        
        # Linhas duplicadas
        duplicate_rows = df.duplicated().sum()
        results['duplicate_rows'] = int(duplicate_rows)
        
        # Linhas completamente vazias
        empty_rows = df.isnull().all(axis=1).sum()
        results['empty_rows'] = int(empty_rows)
        
        # Resumo dos tipos de dados
        for dtype in df.dtypes.unique():
            count = (df.dtypes == dtype).sum()
            results['data_types_summary'][str(dtype)] = int(count)
        
        return results
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calcula score de qualidade geral (0-100)"""
        score = 100.0
        
        # Penalizar issues críticos
        critical_penalty = len(results['critical_issues']) * 15
        score -= critical_penalty
        
        # Penalizar warnings
        warning_penalty = len(results['warnings']) * 3
        score -= warning_penalty
        
        # Penalizar linhas duplicadas
        if results['duplicate_rows'] > 0:
            dup_penalty = min(10, (results['duplicate_rows'] / results['total_records']) * 100)
            score -= dup_penalty
        
        # Penalizar linhas vazias
        if results['empty_rows'] > 0:
            empty_penalty = min(5, (results['empty_rows'] / results['total_records']) * 100)
            score -= empty_penalty
        
        return max(0, round(score, 1))
    
    def _log_validation_results(self, results: Dict[str, Any]):
        """Log dos resultados de validação"""
        score = results['quality_score']
        
        if score >= 90:
            logger.info(f"✅ Qualidade dos dados: {score}% - Excelente!")
        elif score >= 75:
            logger.info(f"✅ Qualidade dos dados: {score}% - Boa")
        elif score >= 50:
            logger.warning(f"⚠️  Qualidade dos dados: {score}% - Regular")
        else:
            logger.error(f"❌ Qualidade dos dados: {score}% - Problemática")
        
        # Log issues críticos
        if results['critical_issues']:
            logger.error("🚨 Issues críticos encontrados:")
            for issue in results['critical_issues']:
                logger.error(f"   • {issue}")
        
        # Log warnings
        if results['warnings']:
            logger.warning("⚠️  Warnings encontrados:")
            for warning in results['warnings'][:5]:  # Mostrar apenas os 5 primeiros
                logger.warning(f"   • {warning}")
            
            if len(results['warnings']) > 5:
                logger.warning(f"   ... e mais {len(results['warnings']) - 5} warnings")
