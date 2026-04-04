#!/usr/bin/env python3
"""
Experiment #4351: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts on 6h aligned with weekly pivot direction (from 1d data) and volume confirmation (>1.5x average) capture institutional breakouts in both bull and bear markets. Weekly pivot provides structural bias (long above weekly pivot, short below). Volume filter ensures breakout conviction. Targets 50-150 total trades over 4 years with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4351_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior week's OHLC
        # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
        weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)  # Prior week high
        weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)    # Prior week low
        weekly_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)  # Prior week close
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Weekly pivot direction: 1 = bullish (price above pivot), -1 = bearish (price below pivot)
        weekly_pivot_direction = np.where(df_1d['close'] > weekly_pivot, 1, -1)
        weekly_pivot_direction_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_direction.values)
    else:
        weekly_pivot_direction_aligned = np.full(n, 0)
    
    # === 6h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(weekly_pivot_direction_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation (> 1.5x average) to filter weak breakouts
        volume_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        bullish_breakout = close[i] > donchian_high[i-1]  # Break above prior period high
        bearish_breakout = close[i] < donchian_low[i-1]   # Break below prior period low
        
        # Align with weekly pivot direction from 1d data
        pivot_bias = weekly_pivot_direction_aligned[i]
        
        # Long conditions: Bullish breakout + volume + bullish weekly pivot bias
        long_entry = bullish_breakout and volume_confirm and (pivot_bias > 0)
        
        # Short conditions: Bearish breakout + volume + bearish weekly pivot bias
        short_entry = bearish_breakout and volume_confirm and (pivot_bias < 0)
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #4351: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts on 6h aligned with weekly pivot direction (from 1d data) and volume confirmation (>1.5x average) capture institutional breakouts in both bull and bear markets. Weekly pivot provides structural bias (long above weekly pivot, short below). Volume filter ensures breakout conviction. Targets 50-150 total trades over 4 years with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4351_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior week's OHLC
        # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
        weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)  # Prior week high
        weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)    # Prior week low
        weekly_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)  # Prior week close
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Weekly pivot direction: 1 = bullish (price above pivot), -1 = bearish (price below pivot)
        weekly_pivot_direction = np.where(df_1d['close'] > weekly_pivot, 1, -1)
        weekly_pivot_direction_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_direction.values)
    else:
        weekly_pivot_direction_aligned = np.full(n, 0)
    
    # === 6h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(weekly_pivot_direction_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation (> 1.5x average) to filter weak breakouts
        volume_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        bullish_breakout = close[i] > donchian_high[i-1]  # Break above prior period high
        bearish_breakout = close[i] < donchian_low[i-1]   # Break below prior period low
        
        # Align with weekly pivot direction from 1d data
        pivot_bias = weekly_pivot_direction_aligned[i]
        
        # Long conditions: Bullish breakout + volume + bullish weekly pivot bias
        long_entry = bullish_breakout and volume_confirm and (pivot_bias > 0)
        
        # Short conditions: Bearish breakout + volume + bearish weekly pivot bias
        short_entry = bearish_breakout and volume_confirm and (pivot_bias < 0)
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals