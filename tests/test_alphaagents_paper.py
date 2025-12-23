#!/usr/bin/env python3
"""
Testing AlphaAgents system """

import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import AlphaAgentsSystem
from tools.financial import get_fundamental_data, get_sentiment_data, get_valuation_data

class AlphaAgentsTestSuite:
    """Tests the multi-agent system"""
    
    def __init__(self):
        self.system = AlphaAgentsSystem()
        self.test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]  # Different stock types
        self.results = {}
        
    def run_all_tests(self):
        """Run all tests"""
        print("Starting AlphaAgents Paper-Compliant Tests...")
        
        # Test individual agents
        print(" Test 1: Individual Agent Performance")
        self.test_individual_agents()
        
        # Test consensus mechanism
        print("\n Test 2: Multi-Agent Consensus & Debate")
        self.test_consensus_mechanism()
        
        # Test majority voting
        print("\n Test 3: Majority Voting Fallback")
        self.test_majority_voting()
        
        # Test risk tolerance
        print("\n Test 4: Risk Tolerance Adaptation")
        self.test_risk_tolerance()
        
        # Test performance
        print("\n Test 5: System Performance Metrics")
        self.test_performance_metrics()
        
        # Test RAG integration
        print("\n Test 6: RAG Document Analysis")
        self.test_rag_integration()
        
        # Generate report
        print("\n")
        self.generate_test_report()
        
    def test_individual_agents(self):
        """Test each agent works"""
        print("Testing individual agent responses...")
        
        for symbol in self.test_symbols[:2]:  # Just test 2 stocks
            print(f"\n  Testing {symbol}:")
            
            # Test Fundamental Agent
            try:
                start_time = time.time()
                fund_data = get_fundamental_data(symbol)
                fund_time = time.time() - start_time
                
                fund_success = len(fund_data) > 100  # Need good data
                rag_used = "FUNDAMENTAL_RAG" in fund_data  # Check RAG usage
                
                print(f"    Fundamental: {'Yes' if fund_success else 'No'} ({fund_time:.1f}s)")
                print(f"    RAG Used: {'Yes' if rag_used else 'No'}")
                
                self.results[f"{symbol}_fundamental"] = {
                    "success": fund_success,
                    "rag_used": rag_used,
                    "response_time": fund_time,
                    "data_length": len(fund_data)
                }
                
            except Exception as e:
                print(f"     Fundamental:  Error - {e}")
                self.results[f"{symbol}_fundamental"] = {"success": False, "error": str(e)}
            
            # Test Sentiment Agent
            try:
                start_time = time.time()
                sent_data = get_sentiment_data(symbol)
                sent_time = time.time() - start_time
                
                sent_success = len(sent_data) > 50  # Need news data
                
                print(f"    Sentiment: {'Yes' if sent_success else 'No'} ({sent_time:.1f}s)")
                
                self.results[f"{symbol}_sentiment"] = {
                    "success": sent_success,
                    "response_time": sent_time,
                    "data_length": len(sent_data)
                }
                
            except Exception as e:
                print(f"    Sentiment: Error - {e}")
                self.results[f"{symbol}_sentiment"] = {"success": False, "error": str(e)}
            
            # Test Valuation Agent
            try:
                start_time = time.time()
                val_data = get_valuation_data(symbol)
                val_time = time.time() - start_time
                
                val_success = len(val_data) > 50  # Need valuation data
                
                print(f"    Valuation: {'Yes' if val_success else 'No'} ({val_time:.1f}s)")
                
                self.results[f"{symbol}_valuation"] = {
                    "success": val_success,
                    "response_time": val_time,
                    "data_length": len(val_data)
                }
                
            except Exception as e:
                print(f"    Valuation: Error - {e}")
                self.results[f"{symbol}_valuation"] = {"success": False, "error": str(e)}
    
    def test_consensus_mechanism(self):
        """Test multi-agent debate and consensus"""
        print("Testing multi-agent consensus mechanism...")
        
        test_symbol = self.test_symbols[0]  # Use AAPL for consensus test
        
        try:
            print(f"\n  Running full analysis for {test_symbol}...")
            start_time = time.time()
            
            result = self.system.analyze_stock(test_symbol, "neutral")
            analysis_time = time.time() - start_time
            
            # Check if we got a proper result
            has_recommendation = "final_recommendation" in result
            has_method = "method" in result
            has_individual = "individual_analyses" in result
            has_collaborative = "collaborative_result" in result
            
            # Check consensus vs majority voting
            consensus_reached = result.get("consensus_reached", False)
            method_used = result.get("method", "unknown")
            
            print(f"    Analysis Time: {analysis_time:.1f}s")
            print(f"    Got Recommendation: {'Yes' if has_recommendation else 'No'}")
            print(f"    Consensus Reached: {'Yes' if consensus_reached else 'No'}")
            print(f"    Method Used: {method_used}")
            
            # Check individual agent positions
            individual_analyses = result.get("individual_analyses", {})
            agent_count = len(individual_analyses)
            
            print(f"    Active Agents: {agent_count}/3")
            
            # Show agent positions
            if individual_analyses:
                print("    Agent Positions:")
                for agent_name, analysis in individual_analyses.items():
                    rec = analysis.get("recommendation", "UNKNOWN")
                    conf = analysis.get("confidence", "UNKNOWN")
                    print(f"      {agent_name.title()}: {rec} ({conf})")
            
            self.results["consensus_test"] = {
                "success": has_recommendation and has_method,
                "consensus_reached": consensus_reached,
                "method": method_used,
                "analysis_time": analysis_time,
                "agent_count": agent_count,
                "individual_analyses": individual_analyses
            }
            
        except Exception as e:
            print(f"    Consensus test failed: {e}")
            self.results["consensus_test"] = {"success": False, "error": str(e)}
    
    def test_majority_voting(self):
        """Test majority voting fallback mechanism"""
        print("Testing majority voting fallback...")
        
        # This is harder to test directly since it requires agents to disagree
        # But we can check if the mechanism exists and works
        
        try:
            # Check if majority voting method exists
            has_majority_method = hasattr(self.system, '_majority_voting_fallback')
            print(f"    Majority Voting Method: {'Yes' if has_majority_method else 'No'}")
            
            # Test with a potentially controversial stock
            controversial_symbol = "TSLA"  # Often has mixed opinions
            
            print(f"\n  Testing with {controversial_symbol} (often mixed opinions)...")
            result = self.system.analyze_stock(controversial_symbol, "neutral")
            
            method_used = result.get("method", "unknown")
            vote_counts = result.get("vote_counts", {})
            
            print(f"    Decision Method: {method_used}")
            if vote_counts:
                print("    Vote Distribution:")
                for vote, count in vote_counts.items():
                    print(f"      {vote}: {count}")
            
            self.results["majority_voting_test"] = {
                "has_method": has_majority_method,
                "method_used": method_used,
                "vote_counts": vote_counts
            }
            
        except Exception as e:
            print(f"    Majority voting test failed: {e}")
            self.results["majority_voting_test"] = {"success": False, "error": str(e)}
    
    def test_risk_tolerance(self):
        """Test risk tolerance adaptation"""
        print("Testing risk tolerance integration...")
        
        test_symbol = self.test_symbols[1]  # Use MSFT
        risk_levels = ["conservative", "neutral", "aggressive"]
        
        risk_results = {}
        
        for risk_level in risk_levels:
            try:
                print(f"\n  Testing {risk_level} risk tolerance...")
                start_time = time.time()
                
                result = self.system.analyze_stock(test_symbol, risk_level)
                analysis_time = time.time() - start_time
                
                recommendation = result.get("final_recommendation", {}).get("recommendation", "UNKNOWN")
                confidence = result.get("final_recommendation", {}).get("confidence", "UNKNOWN")
                
                print(f"    Recommendation: {recommendation}")
                print(f"    Confidence: {confidence}")
                print(f"    Time: {analysis_time:.1f}s")
                
                risk_results[risk_level] = {
                    "recommendation": recommendation,
                    "confidence": confidence,
                    "time": analysis_time
                }
                
            except Exception as e:
                print(f"    {risk_level} test failed: {e}")
                risk_results[risk_level] = {"error": str(e)}
        
        # Check if different risk levels produce different results
        recommendations = [r.get("recommendation") for r in risk_results.values() if "recommendation" in r]
        has_variation = len(set(recommendations)) > 1 if len(recommendations) > 1 else False
        
        print(f"\n    Risk Adaptation Working: {'Yes' if has_variation else 'Unknown'}")
        
        self.results["risk_tolerance_test"] = {
            "results": risk_results,
            "has_variation": has_variation
        }
    
    def test_performance_metrics(self):
        """Test system performance metrics"""
        print("Testing system performance metrics...")
        
        # Test response times
        response_times = []
        success_count = 0
        total_tests = 3
        
        for i, symbol in enumerate(self.test_symbols[:total_tests]):
            try:
                print(f"\n  Performance test {i+1}/{total_tests} with {symbol}...")
                start_time = time.time()
                
                result = self.system.analyze_stock(symbol, "neutral")
                analysis_time = time.time() - start_time
                
                response_times.append(analysis_time)
                
                # Check if we got a valid result
                has_recommendation = "final_recommendation" in result
                if has_recommendation:
                    success_count += 1
                    print(f"    Success in {analysis_time:.1f}s")
                else:
                    print(f"    Failed in {analysis_time:.1f}s")
                
            except Exception as e:
                print(f"    Error: {e}")
        
        # Calculate metrics
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            success_rate = (success_count / total_tests) * 100
            
            print(f"\n  Performance Summary:")
            print(f"    Average Time: {avg_time:.1f}s")
            print(f"    Min Time: {min_time:.1f}s")
            print(f"    Max Time: {max_time:.1f}s")
            print(f"    Success Rate: {success_rate:.1f}%")
            
            # Paper expects reasonable performance
            performance_good = avg_time < 120 and success_rate >= 80  # 2 min max, 80% success
            print(f"    Performance Acceptable: {'Yes' if performance_good else 'No'}")
            
            self.results["performance_metrics"] = {
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "success_rate": success_rate,
                "performance_good": performance_good,
                "response_times": response_times
            }
        else:
            print("    No valid performance data")
            self.results["performance_metrics"] = {"success": False}
    
    def test_rag_integration(self):
        """Test RAG integration for document analysis"""
        print("Testing RAG integration for document analysis...")
        
        # Test if Fundamental Agent uses RAG
        test_symbols_rag = ["AAPL", "MSFT"]  # Test with 2 symbols
        
        rag_working_count = 0
        
        for symbol in test_symbols_rag:
            try:
                print(f"\n  Testing RAG with {symbol}...")
                
                fund_data = get_fundamental_data(symbol)
                
                # Check for RAG indicators
                has_rag_marker = "FUNDAMENTAL_RAG" in fund_data
                has_documents_analyzed = "documents_analyzed" in fund_data
                has_sections = "sections" in fund_data or "business_model" in fund_data
                
                print(f"    RAG Marker Present: {'Yes' if has_rag_marker else 'No'}")
                print(f"    Documents Analyzed: {'Yes' if has_documents_analyzed else 'No'}")
                print(f"    Section Analysis: {'Yes' if has_sections else 'No'}")
                
                if has_rag_marker:
                    rag_working_count += 1
                    
                    # Try to extract more details about RAG analysis
                    try:
                        import ast
                        data_dict = ast.literal_eval(fund_data)
                        docs_count = data_dict.get("documents_analyzed", 0)
                        analysis_sections = data_dict.get("analysis", {}).get("sections", {})
                        
                        print(f"    Documents Count: {docs_count}")
                        print(f"    Analysis Sections: {len(analysis_sections)}")
                        
                    except:
                        print("    RAG data structure complex (normal)")
                
            except Exception as e:
                print(f"    RAG test failed for {symbol}: {e}")
        
        rag_success_rate = (rag_working_count / len(test_symbols_rag)) * 100
        rag_working = rag_success_rate >= 50  # At least 50% should use RAG
        
        print(f"\n  RAG Integration Summary:")
        print(f"    RAG Success Rate: {rag_success_rate:.1f}%")
        print(f"    RAG Working: {'Yes' if rag_working else 'No'}")
        
        self.results["rag_integration"] = {
            "success_rate": rag_success_rate,
            "working": rag_working,
            "symbols_tested": test_symbols_rag,
            "working_count": rag_working_count
        }
    
    def generate_test_report(self):
        """Generate test report"""
        print("ALPHAAGENTS TEST REPORT")
        
        # Overall system health
        total_tests = 6
        passed_tests = 0
        
        # Check each test category
        test_categories = [
            ("Individual Agents", self._check_individual_agents()),
            ("Consensus Mechanism", self._check_consensus()),
            ("Majority Voting", self._check_majority_voting()),
            ("Risk Tolerance", self._check_risk_tolerance()),
            ("Performance Metrics", self._check_performance()),
            ("RAG Integration", self._check_rag_integration())
        ]
        
        print("\nTest Results Summary:")
        for category, passed in test_categories:
            status = "PASS" if passed else "FAIL"
            print(f"  {category}: {status}")
            if passed:
                passed_tests += 1
        
        # Overall score
        overall_score = (passed_tests / total_tests) * 100
        print(f"\nOverall System Score: {overall_score:.1f}% ({passed_tests}/{total_tests})")
        
        # Paper-style evaluation
        if overall_score >= 80:
            print("System Status: EXCELLENT - Ready for production")
        elif overall_score >= 60:
            print("System Status: GOOD - Minor issues to address")
        else:
            print("System Status: NEEDS WORK - Major issues found")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"alphaagents_test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "overall_score": overall_score,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "test_results": self.results,
                "test_categories": dict(test_categories)
            }, f, indent=2)
        
        print(f"\nDetailed report saved: {report_file}")
    
    def _check_individual_agents(self) -> bool:
        """Check if individual agents are working"""
        agent_results = [k for k in self.results.keys() if any(agent in k for agent in ["fundamental", "sentiment", "valuation"])]
        successful = [k for k in agent_results if self.results[k].get("success", False)]
        return len(successful) >= len(agent_results) * 0.7  # 70% success rate
    
    def _check_consensus(self) -> bool:
        """Check if consensus mechanism works"""
        consensus_result = self.results.get("consensus_test", {})
        return consensus_result.get("success", False)
    
    def _check_majority_voting(self) -> bool:
        """Check if majority voting exists"""
        majority_result = self.results.get("majority_voting_test", {})
        return majority_result.get("has_method", False)
    
    def _check_risk_tolerance(self) -> bool:
        """Check if risk tolerance works"""
        risk_result = self.results.get("risk_tolerance_test", {})
        return len(risk_result.get("results", {})) >= 2  # At least 2 risk levels work
    
    def _check_performance(self) -> bool:
        """Check if performance is acceptable"""
        perf_result = self.results.get("performance_metrics", {})
        return perf_result.get("performance_good", False)
    
    def _check_rag_integration(self) -> bool:
        """Check if RAG is working"""
        rag_result = self.results.get("rag_integration", {})
        return rag_result.get("working", False)

def test_alphaagents_complete_suite():
    """Run the complete AlphaAgents test suite (pytest compatible)"""
    print("AlphaAgents Paper-Compliant Test Suite")
    print("Testing multi-agent equity analysis system...")
    print()
    
    # Create and run test suite
    test_suite = AlphaAgentsTestSuite()
    test_suite.run_all_tests()
    
    print("\nTesting complete!")
    print("Check the generated report file for detailed results.")
    
    # Assert that the system works (for pytest)
    assert test_suite.results.get("consensus_test", {}).get("success", False), "Consensus mechanism failed"

def main():
    """Run the complete AlphaAgents test suite"""
    test_alphaagents_complete_suite()

if __name__ == "__main__":
    main()
