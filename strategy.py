#!/usr/bin/env python3
"""
Experiment #4767: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot bias (price > weekly pivot = long bias, price < weekly pivot = short bias) with volume confirmation (>1.5x average) capture strong momentum moves. Weekly pivot provides structural bias that works in both bull and bear markets by identifying key institutional levels. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4767_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: Weekly Pivot Points (using prior week's OHLC) ===
    if len(df_1d) >= 5:
        # Calculate weekly OHLC from daily data
        # We need to group daily data into weeks, but for simplicity we'll use rolling weekly
        # Actually, let's compute proper weekly pivot from 1w data directly
        pass
    else:
        pivot = np.full(len(df_1w), np.nan)
        r1 = np.full(len(df_1w), np.nan)
        s1 = np.full(len(df_1w), np.nan)
        r2 = np.full(len(df_1w), np.nan)
        s2 = np.full(len(df_1w), np.nan)
    
    # Proper weekly pivot from 1w data
    if len(df_1w) >= 1:
        # Weekly pivot: (High + Low + Close) / 3
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        pivot_raw = (weekly_high + weekly_low + weekly_close) / 3.0
        # R1 = 2*Pivot - Low
        r1_raw = 2 * pivot_raw - weekly_low
        # S1 = 2*Pivot - High
        s1_raw = 2 * pivot_raw - weekly_high
        # R2 = Pivot + (High - Low)
        r2_raw = pivot_raw + (weekly_high - weekly_low)
        # S2 = Pivot - (High - Low)
        s2_raw = pivot_raw - (weekly_high - weekly_low)
        
        # Align to 6h timeframe
        pivot_1w = align_htf_to_ltf(prices, df_1w, pivot_raw)
        r1_1w = align_htf_to_ltf(prices, df_1w, r1_raw)
        s1_1w = align_htf_to_ltf(prices, df_1w, s1_raw)
        r2_1w = align_htf_to_ltf(prices, df_1w, r2_raw)
        s2_1w = align_htf_to_ltf(prices, df_1w, s2_raw)
    else:
        pivot_1w = np.full(n, np.nan)
        r1_1w = np.full(n, np.nan)
        s1_1w = np.full(n, np.nan)
        r2_1w = np.full(n, np.nan)
        s2_1w = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(pivot_1w[i]) or np.isnan(r1_1w[i]) or np.isnan(s1_1w[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Determine weekly pivot bias
        # Long bias: price above weekly pivot
        # Short bias: price below weekly pivot
        long_bias = price > pivot_1w[i]
        short_bias = price < pivot_1w[i]
        
        # Donchian breakout conditions with pivot bias
        breakout_long = (price >= high_roll[i]) and long_bias and vol_confirm
        breakout_short = (price <= low_roll[i]) and short_bias and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
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
Experiment #4767: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot bias (price > weekly pivot = long bias, price < weekly pivot = short bias) with volume confirmation (>1.5x average) capture strong momentum moves. Weekly pivot provides structural bias that works in both bull and bear markets by identifying key institutional levels. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4767_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: Weekly Pivot Points ===
    if len(df_1w) >= 1:
        # Weekly pivot: (High + Low + Close) / 3
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        pivot_raw = (weekly_high + weekly_low + weekly_close) / 3.0
        # R1 = 2*Pivot - Low
        r1_raw = 2 * pivot_raw - weekly_low
        # S1 = 2*Pivot - High
        s1_raw = 2 * pivot_raw - weekly_high
        # R2 = Pivot + (High - Low)
        r2_raw = pivot_raw + (weekly_high - weekly_low)
        # S2 = Pivot - (High - Low)
        s2_raw = pivot_raw - (weekly_high - weekly_low)
        
        # Align to 6h timeframe
        pivot_1w = align_htf_to_ltf(prices, df_1w, pivot_raw)
        r1_1w = align_htf_to_ltf(prices, df_1w, r1_raw)
        s1_1w = align_htf_to_ltf(prices, df_1w, s1_raw)
        r2_1w = align_htf_to_ltf(prices, df_1w, r2_raw)
        s2_1w = align_htf_to_ltf(prices, df_1w, s2_raw)
    else:
        pivot_1w = np.full(n, np.nan)
        r1_1w = np.full(n, np.nan)
        s1_1w = np.full(n, np.nan)
        r2_1w = np.full(n, np.nan)
        s2_1w = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(pivot_1w[i]) or np.isnan(r1_1w[i]) or np.isnan(s1_1w[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Determine weekly pivot bias
        # Long bias: price above weekly pivot
        # Short bias: price below weekly pivot
        long_bias = price > pivot_1w[i]
        short_bias = price < pivot_1w[i]
        
        # Donchian breakout conditions with pivot bias
        breakout_long = (price >= high_roll[i]) and long_bias and vol_confirm
        breakout_short = (price <= low_roll[i]) and short_bias and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals