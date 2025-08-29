#!/usr/bin/env python3
"""
LUM3N ORBs V2 - Professional Trading Terminal
Advanced ORB (Opening Range Breakout) system with real-time analytics
"""
import asyncio
import yaml
import sys
import time
from pathlib import Path
from datetime import datetime
from ib_insync import IB
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

# Add data module to path
sys.path.insert(0, str(Path(__file__).parent))

from data.historical import HistoricalDataFetcher
from data.streamer import LiveDataStreamer

def print_lumen_banner():
    """Display the LUM3N ORBs startup banner"""
    colorama.init(autoreset=True)
    
    # Orange color codes
    ORANGE = '\033[38;5;208m'      # Bright orange
    DARK_ORANGE = '\033[38;5;166m' # Dark orange for the "3"
    
    banner = f"""
{ORANGE}╔═════════════════════════════════════════════════════════════════════════════╗
║            ██╗     ██╗   ██╗███╗   ███╗{DARK_ORANGE}██████╗ {ORANGE}███╗   ██╗                   ║
║            ██║     ██║   ██║████╗ ████║{DARK_ORANGE}╚════██╗{ORANGE}████╗  ██║                   ║
║            ██║     ██║   ██║██╔████╔██║{DARK_ORANGE} █████╔╝{ORANGE}██╔██╗ ██║                   ║
║            ██║     ██║   ██║██║╚██╔╝██║{DARK_ORANGE} ╚═══██╗{ORANGE}██║╚██╗██║                   ║
║            ███████╗╚██████╔╝██║ ╚═╝ ██║{DARK_ORANGE}██████╔╝{ORANGE}██║ ╚████║                   ║
║            ╚══════╝ ╚═════╝ ╚═╝     ╚═╝{DARK_ORANGE}╚═════╝ {ORANGE}╚═╝  ╚═══╝                   ║
║                                                                             ║
║                      ██████╗ ██████╗ ██████╗ ███████╗                       ║
║                      ██╔══██╗██╔══██╗██╔══██╗██╔════╝                       ║
║                      ██║  ██║██████╔╝██████╔╝███████╗                       ║
║                      ██║  ██║██╔══██╗██╔══██╗╚════██║                       ║
║                      ██████╔╝██║  ██║██████╔╝███████║                       ║
║                      ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚══════╝                       ║
╚═════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    
    print(banner)
    
    # Animated startup sequence
    startup_messages = [
        f"{Fore.CYAN}Initializing trading engine...",
        f"{Fore.YELLOW}Establishing market data connections...", 
        f"{Fore.GREEN}Loading ORB algorithms...",
        f"{Fore.MAGENTA}Optimizing performance parameters...",
        f"{Fore.WHITE}{Style.BRIGHT}LUM3N ORBs ready for trading!"
    ]
    
    for msg in startup_messages:
        print(f"  {msg}")
        time.sleep(0.5)
    
    print(f"\n{Fore.WHITE}{'='*60}{Style.RESET_ALL}")

def print_system_status():
    """Print system status and configuration"""
    now = datetime.now()
    print(f"{Fore.CYAN}Session Date: {Fore.WHITE}{now.strftime('%Y-%m-%d %H:%M:%S ET')}")
    print(f"{Fore.CYAN}System Mode:  {Fore.GREEN}PRODUCTION")
    print(f"{Fore.CYAN}Strategy:     {Fore.YELLOW}Opening Range Breakout (ORB)")
    print(f"{Fore.CYAN}Timeframes:   {Fore.WHITE}1m, 5m, 4h")
    print(f"{Fore.WHITE}{'='*60}{Style.RESET_ALL}")

async def main():
    """Main execution: Load historical + start streaming"""
    # Display sexy startup banner
    print_lumen_banner()
    print_system_status()
    
    # Load config from script directory
    script_dir = Path(__file__).parent
    config_path = script_dir / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup IB connection
    ib = IB()
    
    # try:
    # Connect to IBKR with progress indicator
    print(f"\n{Fore.CYAN}IBKR Connection")
    print(f"{Fore.WHITE}{'─' * 40}")
    
    with tqdm(total=3, desc=f"{Fore.YELLOW}Connecting", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        pbar.set_description(f"{Fore.YELLOW}Establishing connection")
        time.sleep(0.5)
        pbar.update(1)
        
        await ib.connectAsync(
            config['ibkr_host'], 
            config['ibkr_port'], 
            clientId=config['client_id_start']
        )
        pbar.set_description(f"{Fore.GREEN}Authentication successful")
        pbar.update(1)
        time.sleep(0.3)
        
        pbar.set_description(f"{Fore.GREEN}Connection established")
        pbar.update(1)
    
    print(f"{Fore.GREEN}Connected to IBKR Gateway {config['ibkr_host']}:{config['ibkr_port']}")
    
    # Initialize components
    symbols = config['symbols']
    historical_fetcher = HistoricalDataFetcher(symbols)
    streamer = LiveDataStreamer(symbols, config['buffer_size'])
    
    print(f"\n{Fore.CYAN}Market Symbols: {Fore.WHITE}{', '.join(symbols)}")
    print(f"{Fore.CYAN}Historical Days: {Fore.WHITE}{config['historical_days']}")
    
    # Fetch historical data with progress bars
    print(f"\n{Fore.CYAN}Historical Data Loading")
    print(f"{Fore.WHITE}{'─' * 40}")
    historical_data = await historical_fetcher.fetch_all(
        ib, 
        days=config['historical_days']
    )
    
    # Load historical into streamer
    print(f"\n{Fore.CYAN}Data Integration")
    print(f"{Fore.WHITE}{'─' * 40}")
    
    with tqdm(total=len(symbols), desc=f"{Fore.MAGENTA}Loading buffers", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as load_pbar:
        streamer.load_historical_data(historical_data)
        for _ in symbols:
            time.sleep(0.1)
            load_pbar.update(1)
    
    print(f"{Fore.GREEN}Historical data integrated into streaming buffers")
    
    # Start live streaming 
    print(f"\n{Fore.CYAN}Real-time Streaming")
    print(f"{Fore.WHITE}{'─' * 40}")
    
    if not await streamer.start_streaming(ib):
        print(f"{Fore.RED}Failed to start streaming")
        return
    
    # Live trading status
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.GREEN}{Style.BRIGHT}LIVE MARKET MONITORING ACTIVE{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}")
    print(f"{Fore.YELLOW}Tracking: {Fore.WHITE}{', '.join(symbols)}")
    print(f"{Fore.YELLOW}Timeframes: {Fore.WHITE}1m, 5m, 4h candles")
    print(f"{Fore.YELLOW}Strategy: {Fore.WHITE}Opening Range Breakout Detection")
    print(f"{Fore.CYAN}System Status: {Fore.GREEN}OPERATIONAL")
    print(f"{Fore.WHITE}{'─'*60}")
    print(f"{Fore.RED}Press Ctrl+C to stop monitoring{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'─'*60}")
    
    try:
        while True:
            await asyncio.sleep(1)  # Keep alive
            
            # Optional: Show current data status
            # for symbol in symbols:
            #     df_1m, df_5m, df_4h = streamer.get_dataframes(symbol)
            #     print(f"{symbol}: {len(df_1m)} 1m bars")
                
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Shutdown initiated by user...")
        print(f"{Fore.CYAN}Stopping data streams...")
            
    # except Exception as e:
        # print(f"Error: {e}")
        
    # finally:
        # Cleanup
        # if streamer.streaming:
        #     await streamer.stop_streaming(ib)
        # if ib.isConnected():
        #     ib.disconnect()
        # print("Shutdown complete")

if __name__ == "__main__":
    colorama.init(autoreset=True)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.GREEN}LUM3N ORBs terminated gracefully")
        print(f"{Fore.CYAN} Session ended - All data saved")
        print(f"{Fore.WHITE}Thank you for using LUM3N ORBs Trading System{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED} FATAL SYSTEM ERROR")
        print(f"{Fore.WHITE}{'='*50}")
        print(f"{Fore.RED}Error: {Fore.YELLOW}{e}")
        print(f"{Fore.WHITE}Please check your configuration and try again.")
        print(f"{Fore.CYAN}Contact support if this error persists.{Style.RESET_ALL}")
        sys.exit(1)