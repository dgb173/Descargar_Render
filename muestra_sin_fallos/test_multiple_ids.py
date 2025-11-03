#!/usr/bin/env python3
# Test multiple IDs to see if we can find cases where the AH is different than "-"

import sys
import os
# Add the current directory to Python path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.estudio_scraper import obtener_datos_preview_ligero, obtener_datos_preview_rapido

def test_multiple_ids():
    """
    Test multiple match IDs to see if there are cases where AH values are properly extracted
    """
    # Test IDs from different contexts to find one where data might be available
    test_ids = ["2784338", "2776232"]
    
    for test_id in test_ids:
        print(f"\nTesting with match ID: {test_id}")
        print("="*50)
        
        print("\n--- Light Version ---")
        try:
            result_ligero = obtener_datos_preview_ligero(test_id)
            
            if "error" not in result_ligero:
                if "recent_indirect" in result_ligero:
                    recent_indirect = result_ligero["recent_indirect"]
                    h2h_col3 = recent_indirect.get("h2h_col3")
                    if h2h_col3:
                        print(f"  H2H Rivales (Col3) - AH: {h2h_col3.get('ah', '-')}")
                        print(f"  Score: {h2h_col3.get('score_line', 'N/A')}")
                        print(f"  Date: {h2h_col3.get('date', 'N/A')}")
                    else:
                        print("  No h2h_col3 data found")
                else:
                    print("  No recent_indirect data found")
            else:
                print(f"  Error: {result_ligero['error']}")
        
        except Exception as e:
            print(f"  Exception: {str(e)}")
            
        print("\n--- Fast Version ---")
        try:
            result_rapido = obtener_datos_preview_rapido(test_id)
            
            if "error" not in result_rapido:
                if "recent_indirect" in result_rapido:
                    recent_indirect = result_rapido["recent_indirect"]
                    h2h_col3 = recent_indirect.get("h2h_col3")
                    if h2h_col3:
                        print(f"  H2H Rivales (Col3) - AH: {h2h_col3.get('ah', '-')}")
                        print(f"  Score: {h2h_col3.get('score_line', 'N/A')}")
                        print(f"  Date: {h2h_col3.get('date', 'N/A')}")
                    else:
                        print("  No h2h_col3 data found")
                else:
                    print("  No recent_indirect data found")
            else:
                print(f"  Error: {result_rapido['error']}")
        
        except Exception as e:
            print(f"  Exception: {str(e)}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    test_multiple_ids()