#!/usr/bin/env python3
"""
Experiment #158: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Daily Donchian breakouts capture medium-term trends. Weekly HMA filter ensures alignment with higher timeframe direction, reducing false breakouts in chop. Volume confirmation adds conviction. Works in both bull (breakouts with volume) and bear (failed breakouts reverse) markets by requiring HTF trend alignment. Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_158_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA21 trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA21 on weekly close
    close_1w = df_1w['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False).mean().values
    wma_full = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21w = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
    trend_up_1w = close_1w > hma_21w
    trend_down_1w = close_1w < hma_21w
    
    # Align to daily timeframe
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr_1d = np.zeros(n)
    tr_1d[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1d[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1d Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(trend_up_1w_aligned[i]) or np.isnan(trend_down_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes below midpoint of Donchian channel
                midpoint = (highest_high[i] + lowest_low[i]) / 2
                if close[i] < midpoint:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes above midpoint of Donchian channel
                midpoint = (highest_high[i] + lowest_low[i]) / 2
                if close[i] > midpoint:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band, 1w uptrend, volume spike
        if high[i] > highest_high[i] and trend_up_1w_aligned[i] and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian lower band, 1w downtrend, volume spike
        elif low[i] < lowest_low[i] and trend_down_1w_aligned[i] and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals