"""
MinionsAI v3.1 - Summary Agent
Specialized agent for information synthesis, report generation, and communication.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import re

from .base_agent import BaseAgent
from ..agent_registry import AgentCapability

logger = logging.getLogger(__name__)


class SummaryAgent(BaseAgent):
    """
    Specialized agent for summarization and synthesis tasks.
    """
    
    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        """Initialize the Summary Agent."""
        super().__init__(
            name="Summary Agent",
            description="Specialized in information synthesis, report generation, and communication",
            capabilities=[
                AgentCapability.INFORMATION_SYNTHESIS,
                AgentCapability.NATURAL_LANGUAGE,
                AgentCapability.DOCUMENT_PROCESSING
            ],
            model_name=model_name,
            base_url=base_url
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Summary Agent."""
        return """You are a Summary Agent, specialized in synthesizing information, creating comprehensive reports, and facilitating clear communication.

Your capabilities include:
- Summarizing complex information into clear, concise formats
- Synthesizing data from multiple sources
- Creating executive summaries and reports
- Extracting key insights and findings
- Adapting communication style for different audiences
- Organizing information logically and coherently
- Highlighting important patterns and trends

When creating summaries:
1. Identify the most important information
2. Organize content logically and coherently
3. Use clear, accessible language
4. Highlight key insights and conclusions
5. Maintain accuracy and objectivity
6. Adapt style to the intended audience
7. Include relevant context and background

Always prioritize clarity, accuracy, and usefulness in your summaries and reports."""
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a summary task."""
        try:
            task_type = task_data.get("task_type", "summarize")
            
            logger.info(f"Summary Agent executing {task_type}")
            
            if task_type == "summarize":
                return await self._create_summary(task_data)
            elif task_type == "synthesize":
                return await self._synthesize_information(task_data)
            elif task_type == "create_report":
                return await self._create_report(task_data)
            elif task_type == "extract_insights":
                return await self._extract_insights(task_data)
            elif task_type == "compare_sources":
                return await self._compare_sources(task_data)
            else:
                return await self._general_summary(task_data)
                
        except Exception as e:
            logger.error(f"Error in Summary Agent task execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": ""
            }
    
    async def _create_summary(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of provided content."""
        content = task_data.get("content", "")
        summary_type = task_data.get("summary_type", "standard")
        length = task_data.get("length", "medium")
        audience = task_data.get("audience", "general")
        
        if not content:
            return {
                "success": False,
                "error": "No content provided for summarization"
            }
        
        # Determine summary parameters based on length
        length_params = {
            "brief": "1-2 paragraphs",
            "medium": "3-5 paragraphs", 
            "detailed": "6-10 paragraphs",
            "executive": "1 page executive summary"
        }
        
        target_length = length_params.get(length, "3-5 paragraphs")
        
        summary_prompt = f"""
        Create a {summary_type} summary of the following content for a {audience} audience:
        
        Content to Summarize:
        {content}
        
        Summary Requirements:
        - Length: {target_length}
        - Type: {summary_type}
        - Audience: {audience}
        
        Please provide:
        1. Main points and key information
        2. Important conclusions or findings
        3. Relevant context and background
        4. Clear, accessible language appropriate for the audience
        
        Focus on accuracy, clarity, and usefulness.
        """
        
        summary = await self.generate_response(summary_prompt)
        
        # Analyze summary characteristics
        summary_stats = self._analyze_summary(summary, content)
        
        return {
            "success": True,
            "task_type": "summarize",
            "original_content_length": len(content),
            "summary": summary,
            "summary_stats": summary_stats,
            "summary_type": summary_type,
            "length": length,
            "audience": audience,
            "creation_timestamp": datetime.now().isoformat()
        }
    
    async def _synthesize_information(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize information from multiple sources."""
        sources = task_data.get("sources", [])
        synthesis_goal = task_data.get("goal", "comprehensive_overview")
        focus_areas = task_data.get("focus_areas", [])
        
        if not sources:
            return {
                "success": False,
                "error": "No sources provided for synthesis"
            }
        
        # Format sources for analysis
        formatted_sources = ""
        for i, source in enumerate(sources, 1):
            if isinstance(source, dict):
                title = source.get("title", f"Source {i}")
                content = source.get("content", "")
                formatted_sources += f"\n--- {title} ---\n{content}\n"
            else:
                formatted_sources += f"\n--- Source {i} ---\n{source}\n"
        
        synthesis_prompt = f"""
        Synthesize information from the following sources to create a {synthesis_goal}:
        
        Sources:
        {formatted_sources}
        
        Focus Areas: {', '.join(focus_areas) if focus_areas else 'All relevant aspects'}
        
        Please provide:
        1. Comprehensive synthesis of all sources
        2. Common themes and patterns
        3. Conflicting information and discrepancies
        4. Key insights and conclusions
        5. Gaps in information
        6. Recommendations for further investigation
        
        Create a coherent, well-organized synthesis that combines insights from all sources.
        """
        
        synthesis = await self.generate_response(synthesis_prompt)
        
        # Analyze synthesis quality
        synthesis_analysis = self._analyze_synthesis(synthesis, sources)
        
        return {
            "success": True,
            "task_type": "synthesize",
            "sources_count": len(sources),
            "synthesis": synthesis,
            "synthesis_analysis": synthesis_analysis,
            "synthesis_goal": synthesis_goal,
            "focus_areas": focus_areas,
            "creation_timestamp": datetime.now().isoformat()
        }
    
    async def _create_report(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive report."""
        topic = task_data.get("topic", "")
        data = task_data.get("data", {})
        report_type = task_data.get("report_type", "analytical")
        sections = task_data.get("sections", ["executive_summary", "findings", "conclusions", "recommendations"])
        
        if not topic:
            return {
                "success": False,
                "error": "No topic provided for report creation"
            }
        
        # Format data for inclusion
        formatted_data = ""
        if data:
            formatted_data = f"\nData and Information:\n{self._format_data_for_report(data)}\n"
        
        report_prompt = f"""
        Create a comprehensive {report_type} report on the topic: {topic}
        
        {formatted_data}
        
        Report Structure:
        {', '.join(sections)}
        
        Please provide a well-structured report including:
        1. Executive Summary (key findings and recommendations)
        2. Detailed Findings (analysis and insights)
        3. Supporting Evidence (data and examples)
        4. Conclusions (main takeaways)
        5. Recommendations (actionable next steps)
        6. Appendices (if relevant)
        
        Use professional formatting with clear headings and logical organization.
        """
        
        report = await self.generate_response(report_prompt)
        
        # Extract report structure
        report_structure = self._parse_report_structure(report)
        
        return {
            "success": True,
            "task_type": "create_report",
            "topic": topic,
            "report": report,
            "report_structure": report_structure,
            "report_type": report_type,
            "sections": sections,
            "creation_timestamp": datetime.now().isoformat()
        }
    
    async def _extract_insights(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key insights from information."""
        content = task_data.get("content", "")
        insight_types = task_data.get("types", ["trends", "patterns", "anomalies", "opportunities"])
        
        if not content:
            return {
                "success": False,
                "error": "No content provided for insight extraction"
            }
        
        insights_prompt = f"""
        Extract key insights from the following content, focusing on: {', '.join(insight_types)}
        
        Content:
        {content}
        
        Please identify and explain:
        1. Key trends and patterns
        2. Notable anomalies or outliers
        3. Opportunities and potential actions
        4. Risks and challenges
        5. Relationships and correlations
        6. Implications and significance
        
        Provide clear, actionable insights with supporting evidence.
        """
        
        insights = await self.generate_response(insights_prompt)
        
        # Parse insights into categories
        categorized_insights = self._categorize_insights(insights)
        
        return {
            "success": True,
            "task_type": "extract_insights",
            "content_length": len(content),
            "insights": insights,
            "categorized_insights": categorized_insights,
            "insight_types": insight_types,
            "extraction_timestamp": datetime.now().isoformat()
        }
    
    async def _compare_sources(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple sources and highlight differences."""
        sources = task_data.get("sources", [])
        comparison_criteria = task_data.get("criteria", ["accuracy", "completeness", "perspective", "credibility"])
        
        if len(sources) < 2:
            return {
                "success": False,
                "error": "At least 2 sources required for comparison"
            }
        
        # Format sources for comparison
        formatted_sources = ""
        for i, source in enumerate(sources, 1):
            if isinstance(source, dict):
                title = source.get("title", f"Source {i}")
                content = source.get("content", "")
                formatted_sources += f"\n--- {title} ---\n{content}\n"
            else:
                formatted_sources += f"\n--- Source {i} ---\n{source}\n"
        
        comparison_prompt = f"""
        Compare the following sources based on: {', '.join(comparison_criteria)}
        
        Sources to Compare:
        {formatted_sources}
        
        Please provide:
        1. Side-by-side comparison of key points
        2. Areas of agreement and consensus
        3. Conflicting information and discrepancies
        4. Unique perspectives or information
        5. Quality and credibility assessment
        6. Recommendations for reconciling differences
        
        Create a comprehensive comparison that highlights similarities and differences.
        """
        
        comparison = await self.generate_response(comparison_prompt)
        
        # Analyze comparison results
        comparison_analysis = self._analyze_comparison(comparison, sources)
        
        return {
            "success": True,
            "task_type": "compare_sources",
            "sources_count": len(sources),
            "comparison": comparison,
            "comparison_analysis": comparison_analysis,
            "comparison_criteria": comparison_criteria,
            "comparison_timestamp": datetime.now().isoformat()
        }
    
    async def _general_summary(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general summary requests."""
        description = task_data.get("description", "")
        content = task_data.get("content", "")
        
        if not description and not content:
            return {
                "success": False,
                "error": "No description or content provided"
            }
        
        general_prompt = f"""
        Handle the following summary request:
        
        Request: {description}
        Content: {content}
        
        Provide appropriate summarization or synthesis based on the request.
        """
        
        response = await self.generate_response(general_prompt)
        
        return {
            "success": True,
            "task_type": "general_summary",
            "description": description,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    # Helper methods
    def _analyze_summary(self, summary: str, original: str) -> Dict[str, Any]:
        """Analyze summary characteristics."""
        return {
            "summary_length": len(summary),
            "original_length": len(original),
            "compression_ratio": len(summary) / len(original) if original else 0,
            "word_count": len(summary.split()),
            "sentence_count": len(re.split(r'[.!?]+', summary)),
            "paragraph_count": len(summary.split('\n\n'))
        }
    
    def _analyze_synthesis(self, synthesis: str, sources: List[Any]) -> Dict[str, Any]:
        """Analyze synthesis quality."""
        return {
            "synthesis_length": len(synthesis),
            "sources_integrated": len(sources),
            "word_count": len(synthesis.split()),
            "complexity_score": len(set(synthesis.lower().split())) / len(synthesis.split()) if synthesis else 0
        }
    
    def _format_data_for_report(self, data: Dict[str, Any]) -> str:
        """Format data for inclusion in reports."""
        formatted = ""
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                formatted += f"{key}: {str(value)[:200]}...\n"
            else:
                formatted += f"{key}: {value}\n"
        return formatted
    
    def _parse_report_structure(self, report: str) -> Dict[str, Any]:
        """Parse report to extract structure."""
        structure = {
            "sections": [],
            "headings": [],
            "key_points": []
        }
        
        lines = report.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.isupper() or line.startswith('#') or line.endswith(':')):
                structure["headings"].append(line)
            elif line.startswith(('-', '*', '•')):
                structure["key_points"].append(line[1:].strip())
        
        return structure
    
    def _categorize_insights(self, insights: str) -> Dict[str, List[str]]:
        """Categorize insights by type."""
        categories = {
            "trends": [],
            "patterns": [],
            "opportunities": [],
            "risks": [],
            "recommendations": []
        }
        
        lines = insights.split('\n')
        current_category = "general"
        
        for line in lines:
            line = line.strip().lower()
            if 'trend' in line:
                current_category = "trends"
            elif 'pattern' in line:
                current_category = "patterns"
            elif 'opportunity' in line:
                current_category = "opportunities"
            elif 'risk' in line or 'challenge' in line:
                current_category = "risks"
            elif 'recommend' in line:
                current_category = "recommendations"
            elif line.startswith(('-', '*', '•')) and current_category in categories:
                categories[current_category].append(line[1:].strip())
        
        return categories
    
    def _analyze_comparison(self, comparison: str, sources: List[Any]) -> Dict[str, Any]:
        """Analyze comparison results."""
        return {
            "comparison_length": len(comparison),
            "sources_compared": len(sources),
            "agreements_found": comparison.lower().count("agree") + comparison.lower().count("consensus"),
            "conflicts_found": comparison.lower().count("conflict") + comparison.lower().count("disagree"),
            "unique_perspectives": comparison.lower().count("unique") + comparison.lower().count("different")
        }
    
    async def _initialize_agent(self) -> None:
        """Summary Agent specific initialization."""
        self.add_to_memory("summary_history", [])
        self.add_to_memory("synthesis_templates", {})
        logger.info("Summary Agent initialized with synthesis capabilities")
    
    async def _cleanup_agent(self) -> None:
        """Summary Agent specific cleanup."""
        logger.info("Summary Agent cleanup completed")
