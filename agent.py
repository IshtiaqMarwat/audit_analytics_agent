# ENHANCED AUDIT DATA ANALYSIS AGENT WITH LANGGRAPH ----
# AI-TIP 001 | ADVANCED DATA ANALYSIS AGENT FOR INTERNAL AUDIT & ASSURANCE ----
# GOALS: 
# - Create a comprehensive data analysis agent for audit teams
# - Support various Excel file formats with dynamic schema detection
# - Implement LangGraph sequential workflow
# - Add audit-specific features: anomaly detection, risk assessment, compliance checks

# Libraries
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolExecutor
from typing import TypedDict, List, Dict, Any, Optional
import os
from pathlib import Path
import sys
import yaml
from pprint import pprint
import pandas as pd
import numpy as np
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# SETUP PATHS
PATH_ROOT = 'audit_data_analysis'
current_dir = Path.cwd() / PATH_ROOT
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# ================================
# CONFIGURATION & SETUP
# ================================

class AuditConfig:
    """Configuration class for audit analysis settings"""
    
    # Risk thresholds for audit findings
    RISK_THRESHOLDS = {
        'high_variance': 3.0,  # Standard deviations
        'outlier_percentile': 99.5,
        'duplicate_threshold': 0.05,  # 5% duplicates is concerning
        'missing_data_threshold': 0.1  # 10% missing data
    }
    
    # Audit-specific analysis types
    AUDIT_ANALYSIS_TYPES = [
        'benford_law',
        'duplicate_detection',
        'gap_analysis',
        'trend_analysis',
        'outlier_detection',
        'compliance_check'
    ]

# State definition for LangGraph
class AnalysisState(TypedDict):
    data: pd.DataFrame
    file_path: str
    analysis_type: str
    query: str
    results: Dict[str, Any]
    visualizations: List[str]
    audit_findings: List[Dict[str, Any]]
    risk_assessment: Dict[str, str]
    recommendations: List[str]
    error_messages: List[str]

# ================================
# UTILITY FUNCTIONS
# ================================

class DataLoader:
    """Enhanced data loader supporting various Excel formats"""
    
    @staticmethod
    def load_excel_file(file_path: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load Excel file with automatic sheet detection and metadata extraction
        """
        try:
            # Read Excel file and inspect sheets
            excel_file = pd.ExcelFile(file_path)
            sheets_info = {}
            sheet_rows = {}

            # Analyze each sheet and record row counts
            for sheet_name in excel_file.sheet_names:
                df_temp = pd.read_excel(file_path, sheet_name=sheet_name)
                sheet_rows[sheet_name] = len(df_temp)
                sheets_info[sheet_name] = {
                    'columns': list(df_temp.columns),
                    'sample_data': df_temp.head(2).to_dict('records')
                }

            # Choose the sheet with the most rows as the main sheet
            main_sheet = max(sheet_rows, key=sheet_rows.get)
            df = pd.read_excel(file_path, sheet_name=main_sheet)
            
            # Extract metadata
            metadata = {
                'file_name': Path(file_path).name,
                'sheets': list(excel_file.sheet_names),
                'main_sheet': main_sheet,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'missing_data_summary': df.isnull().sum().to_dict(),
                'sheets_info': sheets_info
            }
            
            return df, metadata
            
        except Exception as e:
            raise Exception(f"Error loading Excel file: {str(e)}")

class AuditAnalyzer:
    """Specialized analyzer for audit and assurance tasks"""
    
    @staticmethod
    def benford_law_analysis(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Perform Benford's Law analysis for fraud detection
        """
        try:
            # Extract first digits
            values = df[column].dropna()
            # Convert to string and extract first digit
            first_digits = values.astype(str).str[0]
            first_digits = first_digits[first_digits.str.isdigit()].astype(int)
            first_digits = first_digits[first_digits != 0]
            
            # Calculate observed frequencies
            observed_freq = first_digits.value_counts().sort_index()
            observed_freq = observed_freq / observed_freq.sum()
            
            # Benford's Law expected frequencies
            expected_freq = pd.Series({
                i: np.log10(1 + 1/i) for i in range(1, 10)
            })
            
            # Calculate chi-square test
            chi_square = sum((observed_freq.get(i, 0) - expected_freq[i])**2 / expected_freq[i] 
                           for i in range(1, 10))
            
            # Risk assessment
            risk_level = "HIGH" if chi_square > 15.507 else "MEDIUM" if chi_square > 11.070 else "LOW"
            
            return {
                'observed_frequencies': observed_freq.to_dict(),
                'expected_frequencies': expected_freq.to_dict(),
                'chi_square_statistic': chi_square,
                'risk_level': risk_level,
                'analysis_summary': f"Benford's Law analysis shows {risk_level} risk of manipulation"
            }
        except Exception as e:
            return {'error': f"Benford analysis failed: {str(e)}"}
    
    @staticmethod
    def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive duplicate detection analysis
        """
        try:
            total_records = len(df)
            if total_records == 0:
                return {
                    'full_duplicates': 0,
                    'partial_duplicates': 0,
                    'duplicate_percentage': 0,
                    'risk_level': 'LOW',
                    'total_records': 0,
                    'analysis_summary': 'No records to analyze'
                }

            # Full row duplicates
            full_duplicates = df.duplicated().sum()

            # Partial duplicates (excluding ID columns)
            id_columns = [col for col in df.columns if 'id' in col.lower() or 'key' in col.lower()]
            non_id_columns = [col for col in df.columns if col not in id_columns]

            if non_id_columns:
                partial_duplicates = df[non_id_columns].duplicated().sum()
            else:
                partial_duplicates = 0

            # Duplicate percentage
            duplicate_percentage = (full_duplicates / total_records) * 100
            
            # Risk assessment
            risk_level = "HIGH" if duplicate_percentage > 5 else "MEDIUM" if duplicate_percentage > 1 else "LOW"
            
            return {
                'full_duplicates': int(full_duplicates),
                'partial_duplicates': int(partial_duplicates),
                'duplicate_percentage': round(duplicate_percentage, 2),
                'risk_level': risk_level,
                'total_records': len(df),
                'analysis_summary': f"Found {full_duplicates} duplicate records ({duplicate_percentage:.2f}%)"
            }
        except Exception as e:
            return {'error': f"Duplicate detection failed: {str(e)}"}
    
    @staticmethod
    def gap_analysis(df: pd.DataFrame, sequence_column: str) -> Dict[str, Any]:
        """
        Identify gaps in sequential data (useful for invoice numbers, etc.)
        """
        try:
            # Convert to numeric and sort
            numeric_values = pd.to_numeric(df[sequence_column], errors='coerce').dropna().sort_values()
            if numeric_values.empty:
                return {
                    'missing_sequences': [],
                    'total_gaps': 0,
                    'gap_percentage': 0,
                    'risk_level': 'LOW',
                    'sequence_range': 'N/A',
                    'analysis_summary': 'No numeric sequence data'
                }
            
            # Find gaps
            gaps = []
            for i in range(1, len(numeric_values)):
                current = numeric_values.iloc[i]
                previous = numeric_values.iloc[i-1]
                if current - previous > 1:
                    gaps.extend(range(int(previous + 1), int(current)))
            
            # Risk assessment
            range_span = numeric_values.max() - numeric_values.min()
            gap_percentage = (len(gaps) / range_span) * 100 if range_span else 0
            risk_level = "HIGH" if gap_percentage > 5 else "MEDIUM" if gap_percentage > 1 else "LOW"
            
            return {
                'missing_sequences': gaps[:50],  # Limit to first 50
                'total_gaps': len(gaps),
                'gap_percentage': round(gap_percentage, 2),
                'risk_level': risk_level,
                'sequence_range': f"{numeric_values.min():.0f} - {numeric_values.max():.0f}",
                'analysis_summary': f"Found {len(gaps)} gaps in sequence ({gap_percentage:.2f}%)"
            }
        except Exception as e:
            return {'error': f"Gap analysis failed: {str(e)}"}

class VisualizationEngine:
    """Enhanced visualization engine for audit reports"""
    
    @staticmethod
    def create_audit_dashboard(df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[go.Figure]:
        """
        Create comprehensive audit dashboard
        """
        figures = []
        
        # 1. Data Quality Overview
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Missing Data by Column', 'Data Types Distribution', 
                          'Record Count Trend', 'Duplicate Analysis'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Missing data
        missing_data = df.isnull().sum()
        fig1.add_trace(
            go.Bar(x=missing_data.index, y=missing_data.values, name="Missing Data"),
            row=1, col=1
        )
        
        # Data types
        dtype_counts = df.dtypes.value_counts()
        fig1.add_trace(
            go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values, name="Data Types"),
            row=1, col=2
        )
        
        fig1.update_layout(title_text="Data Quality Dashboard", showlegend=False)
        figures.append(fig1)
        
        # 2. Benford's Law Analysis (if available)
        if 'benford_analysis' in analysis_results:
            benford_data = analysis_results['benford_analysis']
            if 'observed_frequencies' in benford_data:
                fig2 = go.Figure()
                digits = list(range(1, 10))
                observed = [benford_data['observed_frequencies'].get(i, 0) for i in digits]
                expected = [benford_data['expected_frequencies'].get(i, 0) for i in digits]
                
                fig2.add_trace(go.Bar(x=digits, y=observed, name="Observed", opacity=0.7))
                fig2.add_trace(go.Bar(x=digits, y=expected, name="Expected (Benford's Law)", opacity=0.7))
                fig2.update_layout(
                    title=f"Benford's Law Analysis - Risk Level: {benford_data.get('risk_level', 'Unknown')}",
                    xaxis_title="First Digit",
                    yaxis_title="Frequency"
                )
                figures.append(fig2)
        
        return figures

# ================================
# LANGGRAPH WORKFLOW NODES
# ================================

def load_data_node(state: AnalysisState) -> AnalysisState:
    """Node 1: Load and validate data"""
    try:
        file_path = state['file_path']
        df, metadata = DataLoader.load_excel_file(file_path)
        
        state['data'] = df
        state['results']['metadata'] = metadata
        state['results']['data_loaded'] = True
        
        print(f"✅ Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        
    except Exception as e:
        state['error_messages'].append(f"Data loading failed: {str(e)}")
        
    return state

def analyze_data_node(state: AnalysisState) -> AnalysisState:
    """Node 2: Perform comprehensive data analysis"""
    try:
        df = state['data']
        analysis_type = state['analysis_type']
        
        # Initialize audit analyzer
        analyzer = AuditAnalyzer()
        
        # Perform different types of analysis based on request
        if analysis_type == 'comprehensive' or analysis_type == 'audit':
            # Duplicate detection
            duplicate_results = analyzer.detect_duplicates(df)
            state['results']['duplicate_analysis'] = duplicate_results
            
            # Find numeric columns for Benford's Law
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                # Choose column with highest variance for better detection
                target_column = df[numeric_columns].var().idxmax()
                benford_results = analyzer.benford_law_analysis(df, target_column)
                state['results']['benford_analysis'] = benford_results
            
            # Gap analysis for potential sequence columns
            potential_sequence_cols = [col for col in df.columns 
                                     if any(keyword in col.lower() for keyword in 
                                           ['id', 'number', 'seq', 'invoice', 'receipt', 'order'])]
            
            if potential_sequence_cols:
                gap_results = analyzer.gap_analysis(df, potential_sequence_cols[0])
                state['results']['gap_analysis'] = gap_results
        
        # Statistical analysis
        state['results']['statistical_summary'] = df.describe().to_dict()
        
        print("✅ Data analysis completed")
        
    except Exception as e:
        state['error_messages'].append(f"Data analysis failed: {str(e)}")
        
    return state

def llm_query_node(state: AnalysisState) -> AnalysisState:
    """Node 3: Process natural language queries with LLM"""
    try:
        # Initialize LLM
        model = 'gpt-4o-mini'
        llm = ChatOpenAI(model_name=model, temperature=0)
        
        # Create pandas agent
        df = state['data']
        query = state['query']
        
        # Enhanced suffix for audit context
        audit_suffix = """
        You are an expert data analyst for internal audit and assurance teams. 
        Always return a JSON dictionary that can be parsed into a dataframe containing the requested information.
        Focus on identifying potential risks, anomalies, and compliance issues in your analysis.
        When analyzing financial data, consider materiality thresholds and audit significance.
        """
        
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            agent_type=AgentType.OPENAI_FUNCTIONS,
            suffix=audit_suffix,
            verbose=True,
            allow_dangerous_code=True
        )
        
        # Execute query
        response = agent.invoke(query)
        state['results']['llm_response'] = response['output']
        
        # Try to parse JSON response
        try:
            parsed_data = parse_json_to_dataframe(response['output'])
            state['results']['query_dataframe'] = parsed_data
        except:
            # If parsing fails, store raw response
            state['results']['query_raw_response'] = response['output']
        
        print("✅ LLM query processed")
        
    except Exception as e:
        state['error_messages'].append(f"LLM query failed: {str(e)}")
        
    return state

def risk_assessment_node(state: AnalysisState) -> AnalysisState:
    """Node 4: Perform risk assessment and generate audit findings"""
    try:
        results = state['results']
        findings = []
        
        # Assess duplicate risk
        if 'duplicate_analysis' in results:
            dup_analysis = results['duplicate_analysis']
            if dup_analysis.get('risk_level') in ['HIGH', 'MEDIUM']:
                findings.append({
                    'finding_type': 'Data Quality',
                    'risk_level': dup_analysis['risk_level'],
                    'description': f"High level of duplicate records detected: {dup_analysis['duplicate_percentage']}%",
                    'recommendation': 'Implement data validation controls and investigate duplicate entry procedures'
                })
        
        # Assess Benford's Law risk
        if 'benford_analysis' in results:
            benford_analysis = results['benford_analysis']
            if benford_analysis.get('risk_level') in ['HIGH', 'MEDIUM']:
                findings.append({
                    'finding_type': 'Fraud Risk',
                    'risk_level': benford_analysis['risk_level'],
                    'description': f"Data distribution deviates from Benford's Law: Chi-square = {benford_analysis.get('chi_square_statistic', 'N/A')}",
                    'recommendation': 'Perform detailed transaction testing and investigate potential data manipulation'
                })
        
        # Assess gap analysis risk
        if 'gap_analysis' in results:
            gap_analysis = results['gap_analysis']
            if gap_analysis.get('risk_level') in ['HIGH', 'MEDIUM']:
                findings.append({
                    'finding_type': 'Completeness',
                    'risk_level': gap_analysis['risk_level'],
                    'description': f"Missing sequences detected: {gap_analysis['total_gaps']} gaps ({gap_analysis['gap_percentage']}%)",
                    'recommendation': 'Review sequence numbering controls and investigate missing records'
                })
        
        state['audit_findings'] = findings
        
        # Overall risk assessment
        high_risk_count = sum(1 for f in findings if f['risk_level'] == 'HIGH')
        medium_risk_count = sum(1 for f in findings if f['risk_level'] == 'MEDIUM')
        
        if high_risk_count > 0:
            overall_risk = 'HIGH'
        elif medium_risk_count > 1:
            overall_risk = 'HIGH'
        elif medium_risk_count > 0:
            overall_risk = 'MEDIUM'
        else:
            overall_risk = 'LOW'
        
        state['risk_assessment'] = {
            'overall_risk': overall_risk,
            'high_risk_findings': high_risk_count,
            'medium_risk_findings': medium_risk_count,
            'total_findings': len(findings)
        }
        
        print("✅ Risk assessment completed")
        
    except Exception as e:
        state['error_messages'].append(f"Risk assessment failed: {str(e)}")
        
    return state

def generate_visualizations_node(state: AnalysisState) -> AnalysisState:
    """Node 5: Generate audit-specific visualizations"""
    try:
        df = state['data']
        results = state['results']
        
        viz_engine = VisualizationEngine()
        figures = viz_engine.create_audit_dashboard(df, results)
        
        # Store visualization descriptions
        viz_descriptions = []
        for i, fig in enumerate(figures):
            viz_descriptions.append(f"Visualization {i+1}: {fig.layout.title.text if fig.layout.title else 'Audit Chart'}")
        
        state['visualizations'] = viz_descriptions
        state['results']['visualization_figures'] = figures
        
        print("✅ Visualizations generated")
        
    except Exception as e:
        state['error_messages'].append(f"Visualization generation failed: {str(e)}")
        
    return state

def generate_recommendations_node(state: AnalysisState) -> AnalysisState:
    """Node 6: Generate audit recommendations and final report"""
    try:
        findings = state['audit_findings']
        risk_assessment = state['risk_assessment']
        
        recommendations = []
        
        # Generate recommendations based on findings
        for finding in findings:
            recommendations.append(finding['recommendation'])
        
        # Add general recommendations based on overall risk
        if risk_assessment['overall_risk'] == 'HIGH':
            recommendations.extend([
                'Implement immediate corrective actions for high-risk findings',
                'Establish enhanced monitoring controls',
                'Consider expanding audit scope for related areas'
            ])
        elif risk_assessment['overall_risk'] == 'MEDIUM':
            recommendations.extend([
                'Develop remediation plan with timelines',
                'Implement regular monitoring procedures'
            ])
        
        # Add data quality recommendations
        if 'metadata' in state['results']:
            metadata = state['results']['metadata']
            missing_data_cols = [col for col, count in metadata['missing_data_summary'].items() if count > 0]
            if missing_data_cols:
                recommendations.append(f'Address missing data in columns: {", ".join(missing_data_cols[:5])}')
        
        state['recommendations'] = recommendations
        
        print("✅ Recommendations generated")
        
    except Exception as e:
        state['error_messages'].append(f"Recommendation generation failed: {str(e)}")
        
    return state

# ================================
# UTILITY FUNCTIONS
# ================================

def parse_json_to_dataframe(json_string: str) -> pd.DataFrame:
    """
    Enhanced JSON parser with better error handling
    """
    try:
        # Extract JSON from markdown code blocks
        match = re.search(r'```json\n(.*?)\n```', json_string, re.DOTALL)
        if match:
            json_content = match.group(1)
        else:
            # Try to find JSON without code blocks
            json_content = json_string.strip()
        
        # Parse JSON
        data = json.loads(json_content)
        
        # Handle different JSON structures
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if all(isinstance(v, list) for v in data.values()):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError("Unexpected JSON structure")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Failed to parse JSON to DataFrame: {str(e)}")

# ================================
# MAIN AUDIT ANALYSIS CLASS
# ================================

class AuditDataAnalysisAgent:
    """
    Main class for audit data analysis with LangGraph workflow
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the audit analysis agent
        """
        os.environ['OPENAI_API_KEY'] = openai_api_key
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow
        """
        # Define the workflow
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("load_data", load_data_node)
        workflow.add_node("analyze_data", analyze_data_node)
        workflow.add_node("llm_query", llm_query_node)
        workflow.add_node("risk_assessment", risk_assessment_node)
        workflow.add_node("generate_visualizations", generate_visualizations_node)
        workflow.add_node("generate_recommendations", generate_recommendations_node)
        
        # Define the flow
        workflow.set_entry_point("load_data")
        workflow.add_edge("load_data", "analyze_data")
        workflow.add_edge("analyze_data", "llm_query")
        workflow.add_edge("llm_query", "risk_assessment")
        workflow.add_edge("risk_assessment", "generate_visualizations")
        workflow.add_edge("generate_visualizations", "generate_recommendations")
        workflow.add_edge("generate_recommendations", END)
        
        return workflow.compile()
    
    def analyze_file(self, file_path: str, query: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze an Excel file using the complete workflow
        """
        # Initialize state
        initial_state = AnalysisState(
            data=pd.DataFrame(),
            file_path=file_path,
            analysis_type=analysis_type,
            query=query,
            results={},
            visualizations=[],
            audit_findings=[],
            risk_assessment={},
            recommendations=[],
            error_messages=[]
        )
        
        # Execute workflow
        print("🚀 Starting audit analysis workflow...")
        final_state = self.workflow.invoke(initial_state)
        
        # Prepare final report
        report = {
            'file_analysis': final_state['results'].get('metadata', {}),
            'audit_findings': final_state['audit_findings'],
            'risk_assessment': final_state['risk_assessment'],
            'recommendations': final_state['recommendations'],
            'query_results': final_state['results'].get('query_dataframe', pd.DataFrame()),
            'visualizations': final_state['visualizations'],
            'statistical_analysis': final_state['results'].get('statistical_summary', {}),
            'errors': final_state['error_messages']
        }
        
        return report
    
    def print_audit_report(self, report: Dict[str, Any]):
        """
        Print a formatted audit report
        """
        print("\n" + "="*80)
        print("🔍 INTERNAL AUDIT DATA ANALYSIS REPORT")
        print("="*80)
        
        # File Information
        if 'file_analysis' in report and report['file_analysis']:
            metadata = report['file_analysis']
            print(f"\n📁 FILE INFORMATION:")
            print(f"   File: {metadata.get('file_name', 'Unknown')}")
            print(f"   Records: {metadata.get('total_rows', 0):,}")
            print(f"   Columns: {metadata.get('total_columns', 0)}")
            print(f"   Sheets: {', '.join(metadata.get('sheets', []))}")
        
        # Risk Assessment
        if report['risk_assessment']:
            risk = report['risk_assessment']
            print(f"\n⚠️  OVERALL RISK ASSESSMENT: {risk.get('overall_risk', 'Unknown')}")
            print(f"   High Risk Findings: {risk.get('high_risk_findings', 0)}")
            print(f"   Medium Risk Findings: {risk.get('medium_risk_findings', 0)}")
            print(f"   Total Findings: {risk.get('total_findings', 0)}")
        
        # Audit Findings
        if report['audit_findings']:
            print(f"\n🔍 AUDIT FINDINGS:")
            for i, finding in enumerate(report['audit_findings'], 1):
                print(f"   {i}. [{finding['risk_level']}] {finding['finding_type']}")
                print(f"      {finding['description']}")
        
        # Recommendations
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # Query Results
        if not report['query_results'].empty:
            print(f"\n📊 QUERY RESULTS:")
            print(report['query_results'].to_string(index=False))
        
        # Errors
        if report['errors']:
            print(f"\n❌ ERRORS ENCOUNTERED:")
            for error in report['errors']:
                print(f"   • {error}")
        
        print("\n" + "="*80)

# ================================
# EXAMPLE USAGE & DEMO
# ================================

if __name__ == "__main__":
    # Example usage
    try:
        # Load API key (replace with your method)
        # api_key = yaml.safe_load(open('../credentials.yml'))['openai']
        api_key = "your-openai-api-key-here"  # Replace with actual key
        
        # Initialize the agent
        audit_agent = AuditDataAnalysisAgent(api_key)
        
        # Example file path (replace with actual Excel file)
        file_path = "sample_audit_data.xlsx"
        
        # Example queries for audit teams
        audit_queries = [
            "What are the total transactions by month? Identify any unusual patterns.",
            "Show me the top 10 highest value transactions and flag any potential outliers.",
            "Analyze expense categories and identify any compliance issues.",
            "What are the duplicate entries and what percentage of total data do they represent?"
        ]
        
        # Run analysis
        for query in audit_queries[:1]:  # Run first query as example
            print(f"\n🔎 Analyzing: {query}")
            report = audit_agent.analyze_file(file_path, query, "audit")
            audit_agent.print_audit_report(report)
            
            # Display visualizations if available
            if 'visualization_figures' in report and report['visualization_figures']:
                for fig in report['visualization_figures']:
                    fig.show()
    
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        print("Please ensure you have:")
        print("1. Valid OpenAI API key")
        print("2. Sample Excel file")
        print("3. Required packages installed: pip install langgraph langchain-openai plotly pandas numpy")