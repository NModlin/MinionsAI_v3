"""
MinionsAI v3.1 - Analysis Agent
Specialized agent for data analysis, pattern recognition, and insights generation.
"""

import json
import re
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import statistics

from .base_agent import BaseAgent
from ..agent_registry import AgentCapability

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    """
    Specialized agent for data analysis and pattern recognition tasks.
    """
    
    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        """Initialize the Analysis Agent."""
        super().__init__(
            name="Analysis Agent",
            description="Specialized in data analysis, pattern recognition, and insights generation",
            capabilities=[
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.MATHEMATICAL_COMPUTATION,
                AgentCapability.NATURAL_LANGUAGE,
                AgentCapability.INFORMATION_SYNTHESIS
            ],
            model_name=model_name,
            base_url=base_url
        )
        
        # Analysis-specific configuration
        self.max_data_points = 10000
        self.confidence_threshold = 0.7
        self.pattern_sensitivity = 0.1
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Analysis Agent."""
        return """You are an Analysis Agent, specialized in data analysis, pattern recognition, and generating actionable insights.

Your capabilities include:
- Statistical analysis and data interpretation
- Pattern recognition and trend identification
- Comparative analysis and benchmarking
- Data visualization recommendations
- Predictive insights and forecasting
- Root cause analysis
- Performance metrics evaluation

When analyzing data:
1. Examine data quality and completeness
2. Identify patterns, trends, and anomalies
3. Perform relevant statistical calculations
4. Consider context and external factors
5. Generate actionable insights and recommendations
6. Assess confidence levels and limitations
7. Suggest follow-up analysis if needed

Always be thorough, objective, and provide clear explanations of your analytical methods and findings."""
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an analysis task.
        
        Args:
            task_data: Task data containing analysis parameters
            
        Returns:
            Dict containing analysis results
        """
        try:
            task_type = task_data.get("task_type", "general_analysis")
            data = task_data.get("data", [])
            
            logger.info(f"Analysis Agent executing {task_type}")
            
            if task_type == "statistical_analysis":
                return await self._perform_statistical_analysis(task_data)
            elif task_type == "trend_analysis":
                return await self._perform_trend_analysis(task_data)
            elif task_type == "comparative_analysis":
                return await self._perform_comparative_analysis(task_data)
            elif task_type == "pattern_recognition":
                return await self._perform_pattern_recognition(task_data)
            elif task_type == "text_analysis":
                return await self._perform_text_analysis(task_data)
            elif task_type == "performance_analysis":
                return await self._perform_performance_analysis(task_data)
            else:
                return await self._perform_general_analysis(task_data)
                
        except Exception as e:
            logger.error(f"Error in Analysis Agent task execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": {}
            }
    
    async def _perform_statistical_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis on numerical data."""
        data = task_data.get("data", [])
        if not data:
            return {
                "success": False,
                "error": "No data provided for statistical analysis"
            }
        
        # Convert to numerical data
        numerical_data = self._extract_numerical_data(data)
        if not numerical_data:
            return {
                "success": False,
                "error": "No numerical data found for analysis"
            }
        
        # Calculate basic statistics
        stats = {
            "count": len(numerical_data),
            "mean": statistics.mean(numerical_data),
            "median": statistics.median(numerical_data),
            "mode": statistics.mode(numerical_data) if len(set(numerical_data)) < len(numerical_data) else None,
            "std_dev": statistics.stdev(numerical_data) if len(numerical_data) > 1 else 0,
            "variance": statistics.variance(numerical_data) if len(numerical_data) > 1 else 0,
            "min": min(numerical_data),
            "max": max(numerical_data),
            "range": max(numerical_data) - min(numerical_data)
        }
        
        # Calculate percentiles
        sorted_data = sorted(numerical_data)
        stats["percentiles"] = {
            "25th": self._percentile(sorted_data, 25),
            "50th": self._percentile(sorted_data, 50),
            "75th": self._percentile(sorted_data, 75),
            "90th": self._percentile(sorted_data, 90),
            "95th": self._percentile(sorted_data, 95)
        }
        
        # Generate insights
        insights_prompt = f"""
        Analyze the following statistical data and provide insights:
        
        Statistics:
        {json.dumps(stats, indent=2)}
        
        Provide:
        1. Key statistical insights
        2. Data distribution characteristics
        3. Notable patterns or anomalies
        4. Practical implications
        5. Recommendations for further analysis
        """
        
        insights = await self.generate_response(insights_prompt)
        
        return {
            "success": True,
            "analysis_type": "statistical_analysis",
            "statistics": stats,
            "insights": insights,
            "data_points": len(numerical_data),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _perform_trend_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform trend analysis on time-series data."""
        data = task_data.get("data", [])
        time_field = task_data.get("time_field", "timestamp")
        value_field = task_data.get("value_field", "value")
        
        if not data:
            return {
                "success": False,
                "error": "No data provided for trend analysis"
            }
        
        # Extract time-series data
        time_series = self._extract_time_series(data, time_field, value_field)
        if len(time_series) < 2:
            return {
                "success": False,
                "error": "Insufficient data points for trend analysis (minimum 2 required)"
            }
        
        # Calculate trend metrics
        values = [point["value"] for point in time_series]
        trend_analysis = {
            "data_points": len(time_series),
            "time_span": {
                "start": time_series[0]["time"],
                "end": time_series[-1]["time"]
            },
            "value_range": {
                "min": min(values),
                "max": max(values),
                "change": values[-1] - values[0],
                "percent_change": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
            }
        }
        
        # Calculate moving averages
        if len(values) >= 3:
            trend_analysis["moving_averages"] = {
                "3_period": self._calculate_moving_average(values, 3),
                "5_period": self._calculate_moving_average(values, 5) if len(values) >= 5 else None
            }
        
        # Detect trend direction
        trend_analysis["trend_direction"] = self._detect_trend_direction(values)
        
        # Generate insights
        insights_prompt = f"""
        Analyze the following trend data and provide insights:
        
        Trend Analysis:
        {json.dumps(trend_analysis, indent=2)}
        
        Time Series Data (first 10 points):
        {json.dumps(time_series[:10], indent=2)}
        
        Provide:
        1. Overall trend assessment
        2. Key trend patterns identified
        3. Significant changes or inflection points
        4. Trend strength and consistency
        5. Predictions and recommendations
        """
        
        insights = await self.generate_response(insights_prompt)
        
        return {
            "success": True,
            "analysis_type": "trend_analysis",
            "trend_analysis": trend_analysis,
            "time_series_sample": time_series[:20],  # Include sample for reference
            "insights": insights,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _perform_comparative_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis between datasets."""
        datasets = task_data.get("datasets", {})
        comparison_metrics = task_data.get("metrics", ["mean", "median", "std_dev"])
        
        if len(datasets) < 2:
            return {
                "success": False,
                "error": "At least 2 datasets required for comparative analysis"
            }
        
        # Analyze each dataset
        dataset_stats = {}
        for name, data in datasets.items():
            numerical_data = self._extract_numerical_data(data)
            if numerical_data:
                dataset_stats[name] = {
                    "count": len(numerical_data),
                    "mean": statistics.mean(numerical_data),
                    "median": statistics.median(numerical_data),
                    "std_dev": statistics.stdev(numerical_data) if len(numerical_data) > 1 else 0,
                    "min": min(numerical_data),
                    "max": max(numerical_data)
                }
        
        # Perform comparisons
        comparisons = {}
        dataset_names = list(dataset_stats.keys())
        
        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                name1, name2 = dataset_names[i], dataset_names[j]
                stats1, stats2 = dataset_stats[name1], dataset_stats[name2]
                
                comparison_key = f"{name1}_vs_{name2}"
                comparisons[comparison_key] = {
                    "mean_difference": stats1["mean"] - stats2["mean"],
                    "median_difference": stats1["median"] - stats2["median"],
                    "std_dev_ratio": stats1["std_dev"] / stats2["std_dev"] if stats2["std_dev"] != 0 else None,
                    "range_comparison": {
                        name1: stats1["max"] - stats1["min"],
                        name2: stats2["max"] - stats2["min"]
                    }
                }
        
        # Generate insights
        insights_prompt = f"""
        Analyze the following comparative analysis results:
        
        Dataset Statistics:
        {json.dumps(dataset_stats, indent=2)}
        
        Comparisons:
        {json.dumps(comparisons, indent=2)}
        
        Provide:
        1. Key differences between datasets
        2. Statistical significance of differences
        3. Patterns and relationships identified
        4. Practical implications of the differences
        5. Recommendations based on the comparison
        """
        
        insights = await self.generate_response(insights_prompt)
        
        return {
            "success": True,
            "analysis_type": "comparative_analysis",
            "dataset_statistics": dataset_stats,
            "comparisons": comparisons,
            "insights": insights,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _perform_pattern_recognition(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pattern recognition on data."""
        data = task_data.get("data", [])
        pattern_types = task_data.get("pattern_types", ["frequency", "sequence", "correlation"])
        
        if not data:
            return {
                "success": False,
                "error": "No data provided for pattern recognition"
            }
        
        patterns = {}
        
        # Frequency patterns
        if "frequency" in pattern_types:
            patterns["frequency"] = self._analyze_frequency_patterns(data)
        
        # Sequence patterns
        if "sequence" in pattern_types:
            patterns["sequence"] = self._analyze_sequence_patterns(data)
        
        # Correlation patterns (for numerical data)
        if "correlation" in pattern_types:
            patterns["correlation"] = self._analyze_correlation_patterns(data)
        
        # Generate insights
        insights_prompt = f"""
        Analyze the following pattern recognition results:
        
        Patterns Identified:
        {json.dumps(patterns, indent=2)}
        
        Data Sample:
        {json.dumps(data[:10], indent=2)}
        
        Provide:
        1. Most significant patterns identified
        2. Pattern strength and reliability
        3. Potential causes or explanations
        4. Actionable insights from patterns
        5. Recommendations for pattern utilization
        """
        
        insights = await self.generate_response(insights_prompt)
        
        return {
            "success": True,
            "analysis_type": "pattern_recognition",
            "patterns": patterns,
            "insights": insights,
            "data_points_analyzed": len(data),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _perform_text_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform text analysis on textual data."""
        text_data = task_data.get("text", "")
        analysis_types = task_data.get("analysis_types", ["sentiment", "keywords", "readability"])
        
        if not text_data:
            return {
                "success": False,
                "error": "No text provided for analysis"
            }
        
        # Basic text statistics
        text_stats = {
            "character_count": len(text_data),
            "word_count": len(text_data.split()),
            "sentence_count": len(re.split(r'[.!?]+', text_data)),
            "paragraph_count": len(text_data.split('\n\n'))
        }
        
        # Generate comprehensive text analysis
        analysis_prompt = f"""
        Perform comprehensive text analysis on the following text:
        
        Text Statistics:
        {json.dumps(text_stats, indent=2)}
        
        Text Content:
        {text_data[:1000]}{"..." if len(text_data) > 1000 else ""}
        
        Provide analysis for:
        1. Sentiment analysis (positive/negative/neutral with confidence)
        2. Key themes and topics
        3. Writing style and tone
        4. Readability assessment
        5. Notable patterns or characteristics
        6. Summary and key insights
        """
        
        analysis = await self.generate_response(analysis_prompt)
        
        return {
            "success": True,
            "analysis_type": "text_analysis",
            "text_statistics": text_stats,
            "analysis": analysis,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _perform_performance_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform performance analysis on metrics data."""
        metrics = task_data.get("metrics", {})
        benchmarks = task_data.get("benchmarks", {})
        time_period = task_data.get("time_period", "current")
        
        if not metrics:
            return {
                "success": False,
                "error": "No metrics provided for performance analysis"
            }
        
        # Analyze performance metrics
        performance_analysis = {}
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, (int, float)):
                # Single value metric
                analysis = {"current_value": metric_data}
                
                # Compare with benchmark if available
                if metric_name in benchmarks:
                    benchmark = benchmarks[metric_name]
                    analysis["benchmark_comparison"] = {
                        "benchmark_value": benchmark,
                        "difference": metric_data - benchmark,
                        "percent_difference": ((metric_data - benchmark) / benchmark * 100) if benchmark != 0 else 0,
                        "performance_rating": "above" if metric_data > benchmark else "below" if metric_data < benchmark else "at"
                    }
                
                performance_analysis[metric_name] = analysis
            
            elif isinstance(metric_data, list):
                # Time series metric
                numerical_data = self._extract_numerical_data(metric_data)
                if numerical_data:
                    analysis = {
                        "current_value": numerical_data[-1],
                        "trend": self._detect_trend_direction(numerical_data),
                        "volatility": statistics.stdev(numerical_data) if len(numerical_data) > 1 else 0,
                        "improvement": numerical_data[-1] - numerical_data[0] if len(numerical_data) > 1 else 0
                    }
                    performance_analysis[metric_name] = analysis
        
        # Generate insights
        insights_prompt = f"""
        Analyze the following performance metrics:
        
        Performance Analysis:
        {json.dumps(performance_analysis, indent=2)}
        
        Provide:
        1. Overall performance assessment
        2. Key performance indicators status
        3. Areas of strength and concern
        4. Performance trends and patterns
        5. Actionable recommendations for improvement
        """
        
        insights = await self.generate_response(insights_prompt)
        
        return {
            "success": True,
            "analysis_type": "performance_analysis",
            "performance_analysis": performance_analysis,
            "insights": insights,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _perform_general_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform general analysis on provided data."""
        data = task_data.get("data", [])
        context = task_data.get("context", "")
        
        # Determine data type and perform appropriate analysis
        if isinstance(data, list) and data:
            if all(isinstance(item, (int, float)) for item in data):
                # Numerical data
                return await self._perform_statistical_analysis(task_data)
            elif all(isinstance(item, str) for item in data):
                # Text data
                combined_text = " ".join(data)
                return await self._perform_text_analysis({"text": combined_text})
            else:
                # Mixed data - perform pattern recognition
                return await self._perform_pattern_recognition(task_data)
        
        return {
            "success": False,
            "error": "Unable to determine appropriate analysis method for provided data"
        }
    
    # Helper methods
    def _extract_numerical_data(self, data: List[Any]) -> List[float]:
        """Extract numerical values from mixed data."""
        numerical = []
        for item in data:
            if isinstance(item, (int, float)):
                numerical.append(float(item))
            elif isinstance(item, dict) and "value" in item:
                try:
                    numerical.append(float(item["value"]))
                except (ValueError, TypeError):
                    pass
            elif isinstance(item, str):
                try:
                    numerical.append(float(item))
                except ValueError:
                    pass
        return numerical
    
    def _extract_time_series(self, data: List[Dict], time_field: str, value_field: str) -> List[Dict]:
        """Extract time series data from structured data."""
        time_series = []
        for item in data:
            if isinstance(item, dict) and time_field in item and value_field in item:
                try:
                    time_series.append({
                        "time": item[time_field],
                        "value": float(item[value_field])
                    })
                except (ValueError, TypeError):
                    pass
        return sorted(time_series, key=lambda x: x["time"])
    
    def _percentile(self, sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile of sorted data."""
        if not sorted_data:
            return 0
        
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_data) - 1)
        
        if lower_index == upper_index:
            return sorted_data[lower_index]
        
        weight = index - lower_index
        return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    def _calculate_moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average with specified window."""
        if len(data) < window:
            return []
        
        moving_avg = []
        for i in range(window - 1, len(data)):
            avg = sum(data[i - window + 1:i + 1]) / window
            moving_avg.append(avg)
        
        return moving_avg
    
    def _detect_trend_direction(self, data: List[float]) -> str:
        """Detect overall trend direction in data."""
        if len(data) < 2:
            return "insufficient_data"
        
        # Simple linear trend detection
        increases = 0
        decreases = 0
        
        for i in range(1, len(data)):
            if data[i] > data[i - 1]:
                increases += 1
            elif data[i] < data[i - 1]:
                decreases += 1
        
        if increases > decreases * 1.5:
            return "increasing"
        elif decreases > increases * 1.5:
            return "decreasing"
        else:
            return "stable"
    
    def _analyze_frequency_patterns(self, data: List[Any]) -> Dict[str, Any]:
        """Analyze frequency patterns in data."""
        frequency = {}
        for item in data:
            key = str(item)
            frequency[key] = frequency.get(key, 0) + 1
        
        # Sort by frequency
        sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "most_frequent": sorted_freq[:5],
            "unique_items": len(frequency),
            "total_items": len(data),
            "frequency_distribution": dict(sorted_freq)
        }
    
    def _analyze_sequence_patterns(self, data: List[Any]) -> Dict[str, Any]:
        """Analyze sequence patterns in data."""
        if len(data) < 2:
            return {"error": "Insufficient data for sequence analysis"}
        
        # Look for repeating subsequences
        patterns = {}
        for length in range(2, min(6, len(data) // 2 + 1)):
            for i in range(len(data) - length + 1):
                pattern = tuple(data[i:i + length])
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Filter patterns that occur more than once
        repeating_patterns = {k: v for k, v in patterns.items() if v > 1}
        
        return {
            "repeating_patterns": dict(sorted(repeating_patterns.items(), key=lambda x: x[1], reverse=True)),
            "pattern_count": len(repeating_patterns)
        }
    
    def _analyze_correlation_patterns(self, data: List[Any]) -> Dict[str, Any]:
        """Analyze correlation patterns in numerical data."""
        numerical_data = self._extract_numerical_data(data)
        
        if len(numerical_data) < 3:
            return {"error": "Insufficient numerical data for correlation analysis"}
        
        # Simple autocorrelation analysis
        correlations = {}
        for lag in range(1, min(5, len(numerical_data) // 2)):
            correlation = self._calculate_autocorrelation(numerical_data, lag)
            correlations[f"lag_{lag}"] = correlation
        
        return {
            "autocorrelations": correlations,
            "data_points": len(numerical_data)
        }
    
    def _calculate_autocorrelation(self, data: List[float], lag: int) -> float:
        """Calculate autocorrelation at specified lag."""
        if len(data) <= lag:
            return 0.0
        
        n = len(data) - lag
        mean_val = statistics.mean(data)
        
        numerator = sum((data[i] - mean_val) * (data[i + lag] - mean_val) for i in range(n))
        denominator = sum((x - mean_val) ** 2 for x in data)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    async def _initialize_agent(self) -> None:
        """Analysis Agent specific initialization."""
        self.add_to_memory("analysis_history", [])
        self.add_to_memory("pattern_library", {})
        logger.info("Analysis Agent initialized with analytical capabilities")
    
    async def _cleanup_agent(self) -> None:
        """Analysis Agent specific cleanup."""
        logger.info("Analysis Agent cleanup completed")
