#!/usr/bin/env python3
"""
Experiment #687: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts filtered by weekly pivot levels (using 1d/1w HTF) capture 
institutional order flow. Weekly pivot provides longer-term bias: long when price > weekly R1, 
short when price < weekly S1. Volume confirmation (>2.0x average) ensures breakout validity. 
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 6h timeframe to achieve 
75-200 total trades over 4 years (19-50/year). Works in bull/bear markets via weekly pivot structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_687_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from daily data (using last 5 trading days approximation)
    # For simplicity, use previous day's OHLC for daily pivot, then derive weekly levels
    # In practice, we'll use rolling window of 5 days to approximate weekly
    lookback = 5  # ~1 week of trading days
    if len(high_1d) >= lookback:
        # Weekly high/low/close from past 5 days
        weekly_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
        weekly_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
        weekly_close = pd.Series(close_1d).rolling(window=lookback, min_periods=lookback).last().values
        
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_range = weekly_high - weekly_low
        # Weekly R1 = pivot + (H-L), S1 = pivot - (H-L)
        r1_weekly = weekly_pivot + weekly_range
        s1_weekly = weekly_pivot - weekly_range
    else:
        # Not enough data - use zeros (will be filtered by NaN check)
        weekly_pivot = np.zeros_like(close_1d)
        r1_weekly = np.zeros_like(close_1d)
        s1_weekly = np.zeros_like(close_1d)
    
    # Align weekly pivot levels to 6h timeframe
    r1_weekly_aligned = align_htf_to_ltf(prices, df_1d, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_1d, s1_weekly)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian and weekly pivot calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(r1_weekly_aligned[i]) or
            np.isnan(s1_weekly_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~48h on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up + price above weekly R1 (bullish bias)
            if breakout_up and price > r1_weekly_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down + price below weekly S1 (bearish bias)
            elif breakout_down and price < s1_weekly_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #687: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts filtered by weekly pivot levels (using 1d/1w HTF) capture 
institutional order flow. Weekly pivot provides longer-term bias: long when price > weekly R1, 
short when price < weekly S1. Volume confirmation (>2.0x average) ensures breakout validity. 
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 6h timeframe to achieve 
75-200 total trades over 4 years (19-50/year). Works in bull/bear markets via weekly pivot structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_687_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from daily data (using last 5 trading days approximation)
    # For simplicity, use previous day's OHLC for daily pivot, then derive weekly levels
    # In practice, we'll use rolling window of 5 days to approximate weekly
    lookback = 5  # ~1 week of trading days
    if len(high_1d) >= lookback:
        # Weekly high/low/close from past 5 days
        weekly_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
        weekly_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
        weekly_close = pd.Series(close_1d).rolling(window=lookback, min_periods=lookback).last().values
        
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_range = weekly_high - weekly_low
        # Weekly R1 = pivot + (H-L), S1 = pivot - (H-L)
        r1_weekly = weekly_pivot + weekly_range
        s1_weekly = weekly_pivot - weekly_range
    else:
        # Not enough data - use zeros (will be filtered by NaN check)
        weekly_pivot = np.zeros_like(close_1d)
        r1_weekly = np.zeros_like(close_1d)
        s1_weekly = np.zeros_like(close_1d)
    
    # Align weekly pivot levels to 6h timeframe
    r1_weekly_aligned = align_htf_to_ltf(prices, df_1d, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_1d, s1_weekly)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian and weekly pivot calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(r1_weekly_aligned[i]) or
            np.isnan(s1_weekly_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~48h on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up + price above weekly R1 (bullish bias)
            if breakout_up and price > r1_weekly_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down + price below weekly S1 (bearish bias)
            elif breakout_down and price < s1_weekly_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>