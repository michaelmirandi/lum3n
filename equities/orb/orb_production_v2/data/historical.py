#!/usr/bin/env python3
"""
FAST Historical Data Fetcher
Minimal, efficient historical data loading for ORB indicators
"""
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
from ib_insync import IB, Stock
import pytz
from tqdm import tqdm
from colorama import Fore, Style

class HistoricalDataFetcher:
    """Fast, simple historical data fetcher focused on speed"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.eastern = pytz.timezone('US/Eastern')
        
    async def fetch_all(self, ib: IB, days: int = 5) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch historical data for all symbols and timeframes
        Returns: {symbol: {'1m': df, '5m': df, '4h': df}}
        """
        results = {}
        
        # Progress bar for symbols
        with tqdm(total=len(self.symbols), desc=f"{Fore.CYAN}Symbols", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as symbol_pbar:
            
            for symbol in self.symbols:
                symbol_pbar.set_description(f"{Fore.CYAN}Loading {symbol}")
                results[symbol] = await self._fetch_symbol(ib, symbol, days)
                symbol_pbar.update(1)
        
        print(f"{Fore.GREEN}✅ Historical data fetch complete - {len(results)} symbols loaded")
        return results
    
    async def _fetch_symbol(self, ib: IB, symbol: str, days: int) -> Dict[str, pd.DataFrame]:
        """Fetch all timeframes for one symbol using proven chunked method"""
        
        # Qualify contract (exactly like working version)
        if symbol == 'QQQ':
            contract = Stock(symbol, 'ARCA', 'USD')
        else:
            contract = Stock(symbol, 'SMART', 'USD')
            
        contracts = await ib.qualifyContractsAsync(contract)
        if not contracts:
            raise ValueError(f"Cannot qualify {symbol}")
        contract = contracts[0]
        
        data = {}
        
        # Use proven chunked parameters from working version
        fetch_params = [
            ('1 min', days, 2, '1m'),      # chunks of 2 days
            ('5 mins', days, 3, '5m'),     # chunks of 3 days  
            ('4 hours', 30, 10, '4h')      # 30 days in 10-day chunks
        ]
        
        # Progress bar for timeframes
        timeframe_progress = tqdm(total=len(fetch_params), desc=f"  {Fore.YELLOW}Timeframes", 
                                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}", leave=False)
        
        for bar_size, total_days, chunk_days, key in fetch_params:
            timeframe_progress.set_description(f"  {Fore.YELLOW}Fetching {key} bars")
            
            all_data = []
            current_date = datetime.now()
            num_chunks = (total_days + chunk_days - 1) // chunk_days
            
            # Progress bar for chunks within each timeframe
            chunk_progress = tqdm(total=num_chunks, desc=f"    {Fore.WHITE}Chunks", 
                                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}", leave=False)
            
            for chunk_num in range(num_chunks):
                chunk_end = current_date - timedelta(days=chunk_num * chunk_days)
                end_date_str = chunk_end.strftime('%Y%m%d %H:%M:%S')  # Proper format!
                
                chunk_progress.set_description(f"    {Fore.WHITE}Chunk {chunk_num + 1}")
                
                # try:
                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_date_str,
                    durationStr=f'{chunk_days} D',
                    barSizeSetting=bar_size,
                    whatToShow='TRADES',
                    useRTH=True,  # MARKET HOURS ONLY - key difference!
                    formatDate=1
                )
                
                if bars:
                    chunk_df = self._bars_to_df(bars)
                    if not chunk_df.empty:
                        all_data.append(chunk_df)
                
                chunk_progress.update(1)
                
                # Rate limiting
                if chunk_num < num_chunks - 1:
                    await asyncio.sleep(0.5)
                        
                # except Exception as e:
                #     print(f"    ❌ Chunk {chunk_num} error: {e}")
                #     continue
            
            chunk_progress.close()
            
            # Combine chunks
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                data[key] = combined_df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
                timeframe_progress.write(f"    {Fore.GREEN}✅ {key}: {len(data[key])} bars loaded")
            else:
                data[key] = pd.DataFrame()
                timeframe_progress.write(f"    {Fore.RED}❌ {key}: No data")
                
            timeframe_progress.update(1)
        
        timeframe_progress.close()
        
        return data
    
    def _bars_to_df(self, bars: List) -> pd.DataFrame:
        """Convert IB bars to DataFrame with robust datetime handling"""
        if not bars:
            return pd.DataFrame()

        data_list = []
        for bar in bars:
            # Robust datetime handling (from working cache.py)
            bar_date = bar.date

            if bar_date is None:  # Handle None dates
                continue

            if isinstance(bar_date, str):
                try:
                    if ' ' in bar_date:
                        parsed_date = pd.to_datetime(bar_date, format='%Y%m%d %H:%M:%S')
                    else:
                        parsed_date = pd.to_datetime(bar_date, format='%Y%m%d')
                except:
                    parsed_date = pd.to_datetime(bar_date)
            else:
                parsed_date = pd.Timestamp(bar_date)

            # Timezone handling
            if parsed_date.tz is not None:
                if parsed_date.tz != self.eastern:
                    parsed_date = parsed_date.tz_convert(self.eastern)
                parsed_date = parsed_date.tz_localize(None)

            data_list.append({
                'date': parsed_date,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': float(bar.volume)
            })

        if data_list:
            df = pd.DataFrame(data_list)
            return df.sort_values('date').reset_index(drop=True)
        else:
            return pd.DataFrame()