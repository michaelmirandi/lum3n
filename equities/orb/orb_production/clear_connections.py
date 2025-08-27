#!/usr/bin/env python3
"""
Clear stuck IBKR connections
Use this if you're getting "client id already in use" errors
"""
import asyncio
import sys
from pathlib import Path
from ib_insync import IB

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def clear_connections(host='127.0.0.1', port=7496, start_id=100, end_id=110):
    """
    Try to connect and disconnect multiple client IDs to clear stuck connections
    """
    print("🔧 Clearing potential stuck IBKR connections...")
    print(f"Testing client IDs {start_id} to {end_id}")
    print("-" * 50)
    
    cleared_count = 0
    
    for client_id in range(start_id, end_id + 1):
        ib = IB()
        try:
            print(f"Testing client ID {client_id}...", end=" ")
            
            # Try to connect with short timeout
            await asyncio.wait_for(
                ib.connectAsync(host, port, clientId=client_id),
                timeout=2.0
            )
            
            # If we got here, connection worked
            print("✅ Connected - disconnecting to clear")
            ib.disconnect()
            cleared_count += 1
            await asyncio.sleep(0.1)  # Brief pause
            
        except asyncio.TimeoutError:
            print("⏱️ Timeout (likely in use)")
        except Exception as e:
            if "already in use" in str(e).lower():
                print("🔒 In use (as expected)")
            else:
                print(f"❌ Error: {e}")
        
        # Make sure we disconnect
        try:
            if ib.isConnected():
                ib.disconnect()
        except:
            pass
    
    print("-" * 50)
    print(f"✅ Cleared {cleared_count} connections")
    print("You should now be able to run test_phase1.py")

async def main():
    """Main function"""
    print("="*60)
    print("IBKR CONNECTION CLEANER")
    print("="*60)
    
    try:
        await clear_connections()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⛔ Stopped by user")
        sys.exit(1)