#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Spike

HYPOTHESIS: On daily timeframe, Donchian(20) breakouts aligned with 1-week HMA trend 
and volume confirmation capture strong momentum moves in BTC/ETH/SOL. The weekly HMA 
filters for primary trend direction, reducing false breakouts in choppy markets. 
Designed for low trade frequency (target: 30-100 total over 4 years) to minimize 
fee drag while maintaining edge in both bull and bear markets via trend-following 
logic that works in up trends and avoids shorts in strong downtrends.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(arr, period):
    """Hull Moving Average - reduces lag while maintaining smoothness."""
    n = len(arr)
    if n < period:
        return np.full(n, np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    
    # WMA helper
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = wma(arr, half)
    wma_full = wma(arr, period)
    if len(wma_half) == 0 or len(wma_full) == 0:
        return np.full(n, np.nan)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt)
    
    # Pad to original length
    result = np.full(n, np.nan)
    start_idx = period - len(hma)
    if start_idx >= 0:
        result[start_idx:] = hma
    return result

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w OHLC for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)  # auto shift(1)
    
    # === 1d Indicators ===
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Levels ---
        dc_upper = dc_upper_20[i]
        dc_lower = dc_lower_20[i]
        hma_1w_val = hma_1w_aligned[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~3 days)
            if min_hold:
                if position_side > 0:
                    # Exit long: price below weekly HMA OR touches lower Donchian
                    if close[i] < hma_1w_val or close[i] <= dc_lower:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price above weekly HMA OR touches upper Donchian
                    if close[i] > hma_1w_val or close[i] >= dc_upper:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # 1. Breakout above upper Donchian with price above weekly HMA AND volume confirmation
        # 2. Pullback to lower Donchian in strong uptrend (price > weekly HMA)
        if (close[i] > dc_upper and close[i] > hma_1w_val and vol_ok) or \
           (close[i] <= dc_lower and close[i] > hma_1w_val * 1.02):  # Pullback in uptrend
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # 1. Breakout below lower Donchian with price below weekly HMA AND volume confirmation
        # 2. Rally to upper Donchian in strong downtrend (price < weekly HMA)
        elif (close[i] < dc_lower and close[i] < hma_1w_val and vol_ok) or \
             (close[i] >= dc_upper and close[i] < hma_1w_val * 0.98):  # Rally in downtrend
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals