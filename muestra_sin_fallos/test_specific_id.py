#!/usr/bin/env python3
# Test with the specific ID provided to verify the fix works

import sys
import os
# Add the current directory to Python path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.estudio_scraper import obtener_datos_preview_ligero, obtener_datos_preview_rapido

def test_specific_match_id():
    """
    Test with the specific match ID to verify the fix works correctly
    """
    test_id = "2784338"
    
    print(f"Testing with match ID: {test_id}")
    print("="*50)
    
    print("\n--- Testing obtener_datos_preview_ligero ---")
    try:
        result_ligero = obtener_datos_preview_ligero(test_id)
        
        if "error" in result_ligero:
            print(f"Error in light version: {result_ligero['error']}")
        else:
            print("Light version succeeded")
            
            if "recent_indirect" in result_ligero:
                recent_indirect = result_ligero["recent_indirect"]
                
                if "h2h_col3" in recent_indirect and recent_indirect["h2h_col3"] is not None:
                    h2h_col3 = recent_indirect["h2h_col3"]
                    ah_value = h2h_col3.get("ah", "-")
                    print(f"  H2H Rivales (Col3) - AH: {ah_value}")
                    print(f"  Score: {h2h_col3.get('score_line', 'N/A')}")
                    print(f"  Date: {h2h_col3.get('date', 'N/A')}")
                    
                    if ah_value != "-":
                        print("  [SUCCESS] AH value is properly extracted (not '-')")
                    else:
                        print("  [ISSUE] AH value is still '-'")
                else:
                    print("  [INFO] No 'h2h_col3' data found")
                    
                print(f"  Last Home AH: {recent_indirect.get('last_home', {}).get('ah', '-') if recent_indirect.get('last_home') else 'N/A'}")
                print(f"  Last Away AH: {recent_indirect.get('last_away', {}).get('ah', '-') if recent_indirect.get('last_away') else 'N/A'}")
            else:
                print("  [ERROR] No 'recent_indirect' data found")
    
    except Exception as e:
        print(f"  [ERROR] Exception in light version: {str(e)}")
    
    print("\n--- Testing obtener_datos_preview_rapido ---")
    try:
        result_rapido = obtener_datos_preview_rapido(test_id)
        
        if "error" in result_rapido:
            print(f"Error in fast version: {result_rapido['error']}")
        else:
            print("Fast version succeeded")
            
            if "recent_indirect" in result_rapido:
                recent_indirect = result_rapido["recent_indirect"]
                
                if "h2h_col3" in recent_indirect and recent_indirect["h2h_col3"] is not None:
                    h2h_col3 = recent_indirect["h2h_col3"]
                    ah_value = h2h_col3.get("ah", "-")
                    print(f"  H2H Rivales (Col3) - AH: {ah_value}")
                    print(f"  Score: {h2h_col3.get('score_line', 'N/A')}")
                    print(f"  Date: {h2h_col3.get('date', 'N/A')}")
                    
                    if ah_value != "-":
                        print("  [SUCCESS] AH value is properly extracted (not '-')")
                    else:
                        print("  [ISSUE] AH value is still '-'")
                else:
                    print("  [INFO] No 'h2h_col3' data found")
                    
                print(f"  Last Home AH: {recent_indirect.get('last_home', {}).get('ah', '-') if recent_indirect.get('last_home') else 'N/A'}")
                print(f"  Last Away AH: {recent_indirect.get('last_away', {}).get('ah', '-') if recent_indirect.get('last_away') else 'N/A'}")
            else:
                print("  [ERROR] No 'recent_indirect' data found")
    
    except Exception as e:
        print(f"  [ERROR] Exception in fast version: {str(e)}")

    print("\n" + "="*50)
    print("Comparison:")
    print("If both versions show the same AH values for 'H2H Rivales (Col3)', the fix is working correctly!")

if __name__ == "__main__":
    test_specific_match_id()