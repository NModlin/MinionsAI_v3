"""
MinionsAI v3.1 - Research Agent
Specialized agent for web research, information gathering, and fact verification.
"""

import asyncio
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from duckduckgo_search import DDGS

from .base_agent import BaseAgent
from ..agent_registry import AgentCapability

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Specialized agent for research and information gathering tasks.
    """
    
    def __init__(self, model_name: str = "llama3:8b", base_url: str = "http://localhost:11434"):
        """Initialize the Research Agent."""
        super().__init__(
            name="Research Agent",
            description="Specialized in web research, information gathering, and fact verification",
            capabilities=[
                AgentCapability.WEB_SEARCH,
                AgentCapability.INFORMATION_SYNTHESIS,
                AgentCapability.NATURAL_LANGUAGE,
                AgentCapability.DOCUMENT_PROCESSING
            ],
            model_name=model_name,
            base_url=base_url
        )
        
        # Research-specific configuration
        self.max_search_results = 10
        self.search_timeout = 30
        self.fact_check_sources = 3
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Research Agent."""
        return """You are a Research Agent, specialized in gathering, analyzing, and synthesizing information from various sources.

Your capabilities include:
- Conducting comprehensive web searches
- Fact-checking and verification
- Synthesizing information from multiple sources
- Identifying reliable and authoritative sources
- Extracting key insights and findings
- Organizing research results clearly

When conducting research:
1. Use multiple search queries to get comprehensive coverage
2. Evaluate source credibility and reliability
3. Cross-reference information across sources
4. Identify potential biases or limitations
5. Provide clear, well-organized summaries
6. Include source citations and references

Always be thorough, accurate, and objective in your research approach."""
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a research task.
        
        Args:
            task_data: Task data containing research parameters
            
        Returns:
            Dict containing research results
        """
        try:
            task_type = task_data.get("task_type", "general_research")
            query = task_data.get("query", "")
            max_results = task_data.get("max_results", self.max_search_results)
            
            if not query:
                return {
                    "success": False,
                    "error": "No research query provided",
                    "results": []
                }
            
            logger.info(f"Research Agent executing {task_type} for query: {query}")
            
            if task_type == "web_search":
                return await self._perform_web_search(query, max_results)
            elif task_type == "fact_check":
                return await self._perform_fact_check(task_data)
            elif task_type == "comprehensive_research":
                return await self._perform_comprehensive_research(task_data)
            elif task_type == "source_verification":
                return await self._verify_sources(task_data)
            else:
                return await self._perform_general_research(query, max_results)
                
        except Exception as e:
            logger.error(f"Error in Research Agent task execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    async def _perform_web_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Perform a web search using DuckDuckGo."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            if not results:
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "message": "No search results found"
                }
            
            # Process and enhance results
            processed_results = []
            for i, result in enumerate(results, 1):
                processed_result = {
                    "rank": i,
                    "title": result.get("title", "No title"),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "No description"),
                    "relevance_score": self._calculate_relevance(query, result)
                }
                processed_results.append(processed_result)
            
            # Sort by relevance
            processed_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "success": True,
                "query": query,
                "results": processed_results,
                "total_results": len(processed_results),
                "search_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }
    
    async def _perform_fact_check(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fact-checking on a given statement."""
        statement = task_data.get("statement", "")
        if not statement:
            return {
                "success": False,
                "error": "No statement provided for fact-checking"
            }
        
        # Generate search queries for fact-checking
        fact_check_queries = [
            f"{statement} fact check",
            f"{statement} verify",
            f"{statement} true false",
            f"evidence {statement}"
        ]
        
        all_results = []
        for query in fact_check_queries:
            search_result = await self._perform_web_search(query, 3)
            if search_result["success"]:
                all_results.extend(search_result["results"])
        
        # Analyze results for fact-checking
        analysis_prompt = f"""
        Analyze the following search results to fact-check this statement: "{statement}"
        
        Search Results:
        {self._format_results_for_analysis(all_results)}
        
        Provide a fact-check analysis including:
        1. Verification status (True/False/Partially True/Unverified)
        2. Supporting evidence
        3. Contradicting evidence
        4. Source reliability assessment
        5. Confidence level (1-10)
        """
        
        analysis = await self.generate_response(analysis_prompt)
        
        return {
            "success": True,
            "statement": statement,
            "analysis": analysis,
            "supporting_sources": all_results,
            "fact_check_timestamp": datetime.now().isoformat()
        }
    
    async def _perform_comprehensive_research(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive research on a topic."""
        topic = task_data.get("topic", "")
        research_depth = task_data.get("depth", "standard")  # basic, standard, deep
        
        if not topic:
            return {
                "success": False,
                "error": "No research topic provided"
            }
        
        # Generate multiple research angles
        research_queries = [
            f"{topic} overview",
            f"{topic} latest developments",
            f"{topic} expert opinions",
            f"{topic} statistics data",
            f"{topic} pros and cons"
        ]
        
        if research_depth == "deep":
            research_queries.extend([
                f"{topic} academic research",
                f"{topic} case studies",
                f"{topic} future trends",
                f"{topic} challenges problems"
            ])
        
        # Collect research from all angles
        research_results = {}
        for query in research_queries:
            search_result = await self._perform_web_search(query, 5)
            if search_result["success"]:
                research_results[query] = search_result["results"]
        
        # Synthesize research findings
        synthesis_prompt = f"""
        Synthesize comprehensive research findings on the topic: "{topic}"
        
        Research Data:
        {self._format_research_data(research_results)}
        
        Provide a comprehensive research report including:
        1. Executive Summary
        2. Key Findings
        3. Current State Analysis
        4. Trends and Developments
        5. Expert Perspectives
        6. Data and Statistics
        7. Challenges and Opportunities
        8. Conclusions and Recommendations
        9. Source References
        """
        
        synthesis = await self.generate_response(synthesis_prompt)
        
        return {
            "success": True,
            "topic": topic,
            "research_depth": research_depth,
            "synthesis": synthesis,
            "raw_research_data": research_results,
            "research_timestamp": datetime.now().isoformat()
        }
    
    async def _verify_sources(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the credibility and reliability of sources."""
        sources = task_data.get("sources", [])
        if not sources:
            return {
                "success": False,
                "error": "No sources provided for verification"
            }
        
        verified_sources = []
        for source in sources:
            verification = await self._verify_single_source(source)
            verified_sources.append(verification)
        
        return {
            "success": True,
            "verified_sources": verified_sources,
            "verification_timestamp": datetime.now().isoformat()
        }
    
    async def _verify_single_source(self, source: str) -> Dict[str, Any]:
        """Verify a single source for credibility."""
        # Extract domain and analyze
        verification_query = f"site reliability credibility {source}"
        search_result = await self._perform_web_search(verification_query, 3)
        
        verification_prompt = f"""
        Assess the credibility and reliability of this source: {source}
        
        Search Results:
        {self._format_results_for_analysis(search_result.get("results", []))}
        
        Provide an assessment including:
        1. Source type (academic, news, blog, government, etc.)
        2. Credibility score (1-10)
        3. Reliability factors
        4. Potential biases
        5. Recommendation for use
        """
        
        assessment = await self.generate_response(verification_prompt)
        
        return {
            "source": source,
            "assessment": assessment,
            "search_results": search_result.get("results", [])
        }
    
    async def _perform_general_research(self, query: str, max_results: int) -> Dict[str, Any]:
        """Perform general research with analysis."""
        search_result = await self._perform_web_search(query, max_results)
        
        if not search_result["success"]:
            return search_result
        
        # Generate analysis of search results
        analysis_prompt = f"""
        Analyze and summarize the following search results for the query: "{query}"
        
        Search Results:
        {self._format_results_for_analysis(search_result["results"])}
        
        Provide:
        1. Key findings summary
        2. Main themes and patterns
        3. Notable insights
        4. Source quality assessment
        5. Recommendations for further research
        """
        
        analysis = await self.generate_response(analysis_prompt)
        
        return {
            "success": True,
            "query": query,
            "results": search_result["results"],
            "analysis": analysis,
            "research_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_relevance(self, query: str, result: Dict[str, Any]) -> float:
        """Calculate relevance score for a search result."""
        query_terms = query.lower().split()
        title = result.get("title", "").lower()
        snippet = result.get("body", "").lower()
        
        score = 0.0
        
        # Check title matches
        for term in query_terms:
            if term in title:
                score += 2.0
            if term in snippet:
                score += 1.0
        
        # Normalize by query length
        if query_terms:
            score = score / len(query_terms)
        
        return min(score, 10.0)  # Cap at 10
    
    def _format_results_for_analysis(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM analysis."""
        formatted = ""
        for i, result in enumerate(results, 1):
            formatted += f"\n{i}. {result.get('title', 'No title')}\n"
            formatted += f"   URL: {result.get('url', 'No URL')}\n"
            formatted += f"   Snippet: {result.get('snippet', 'No description')}\n"
        return formatted
    
    def _format_research_data(self, research_data: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format research data for synthesis."""
        formatted = ""
        for query, results in research_data.items():
            formatted += f"\n--- {query} ---\n"
            formatted += self._format_results_for_analysis(results)
        return formatted
    
    async def _initialize_agent(self) -> None:
        """Research Agent specific initialization."""
        self.add_to_memory("search_history", [])
        self.add_to_memory("fact_check_history", [])
        logger.info("Research Agent initialized with search capabilities")
    
    async def _cleanup_agent(self) -> None:
        """Research Agent specific cleanup."""
        # Save research history if needed
        logger.info("Research Agent cleanup completed")
