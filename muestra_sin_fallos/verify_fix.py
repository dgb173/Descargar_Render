#!/usr/bin/env python3
# Simple test to verify the fix is working correctly

from modules.estudio_scraper import obtener_datos_preview_ligero

def test_preview_fix():
    """
    Test function to verify that the fix correctly retrieves AH values
    for the 'Rivales Comunes' section in the preview data.
    """
    print("Testing the fix for AH values in 'Rivales Comunes' section...")
    
    # Use a test ID to see if the function works correctly
    # Note: This might fail if the match ID doesn't exist or network is down
    # but it will help us see if the function structure is correct
    test_match_ids = ["2776232", "123456", "100000"]  # Some test IDs
    
    for test_id in test_match_ids:
        print(f"\n--- Testing with match ID: {test_id} ---")
        try:
            result = obtener_datos_preview_ligero(test_id)
            
            if "error" in result:
                print(f"Error for match ID {test_id}: {result['error']}")
                continue
                
            print(f"Success for match ID {test_id}")
            
            # Check if recent_indirect structure exists
            if "recent_indirect" in result:
                recent_indirect = result["recent_indirect"]
                
                # Check for h2h_col3 (Rivales Comunes)
                if "h2h_col3" in recent_indirect and recent_indirect["h2h_col3"] is not None:
                    h2h_col3 = recent_indirect["h2h_col3"]
                    ah_value = h2h_col3.get("ah", "-")
                    print(f"  H2H Rivales (Col3) - AH: {ah_value}")
                    print(f"  Score: {h2h_col3.get('score_line', 'N/A')}")
                    
                    # This is the critical check: we want to see a real AH value, not "-"
                    if ah_value != "-":
                        print("  [SUCCESS] AH value is properly extracted (not '-')")
                    else:
                        print("  [ISSUE] AH value is still '-'")
                else:
                    print("  ℹ️  No 'h2h_col3' data found for this match")
                    
                # Also check other sections for completeness
                if "last_home" in recent_indirect and recent_indirect["last_home"]:
                    last_home = recent_indirect["last_home"]
                    print(f"  Last Home AH: {last_home.get('ah', '-')}")
                    
                if "last_away" in recent_indirect and recent_indirect["last_away"]:
                    last_away = recent_indirect["last_away"]
                    print(f"  Last Away AH: {last_away.get('ah', '-')}")
            else:
                print("  [ERROR] No 'recent_indirect' data found in result")
        
        except Exception as e:
            print(f"  [ERROR] Exception occurred for match ID {test_id}: {str(e)}")
    
    print(f"\n--- Test completed ---")
    print("If AH values in 'H2H Rivales (Col3)' are not '-', the fix is working correctly!")

if __name__ == "__main__":
    test_preview_fix()