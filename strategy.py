#!/usr/bin/env python3
"""
Experiment #174: 1h Session Filter + 4h/1d Donchian Breakout with Volume Spike

HYPOTHESIS: Trade 1h timeframe only during high-liquidity UTC 08-20 session.
Use 4h Donchian(20) and 1d Donchian(20) for confluence: long when price breaks above both,
short when breaks below both. Require volume > 1.5x 20-bar average. Discrete position size
0.20 to limit drawdown. ATR-based stop at 2.5x ATR. Target 15-37 trades/year on 1h to
avoid fee drag while capturing structured breakouts in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_session_donchian_4h_1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h and 1d Donchian channels (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h Donchian(20)
    dc_upper_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 1d Donchian(20)
    dc_upper_1d = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_1d = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align to 1h timeframe with shift(1) for completed bars only
    dc_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_upper_4h)
    dc_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_lower_4h)
    dc_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, dc_upper_1d)
    dc_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, dc_lower_1d)
    
    # === 1h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: UTC 08-20 ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_4h_aligned[i]) or np.isnan(dc_lower_4h_aligned[i]) or
            np.isnan(dc_upper_1d_aligned[i]) or np.isnan(dc_lower_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Confluence Breakout ---
        bullish_confluence = (close[i] > dc_upper_4h_aligned[i]) and (close[i] > dc_upper_1d_aligned[i])
        bearish_confluence = (close[i] < dc_lower_4h_aligned[i]) and (close[i] < dc_lower_1d_aligned[i])
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False
        
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
            
            # Exit conditions: opposite Donchian touch on either timeframe
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~3h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian on 4h OR 1d
                    if close[i] <= dc_lower_4h_aligned[i] or close[i] <= dc_lower_1d_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian on 4h OR 1d
                    if close[i] >= dc_upper_4h_aligned[i] or close[i] >= dc_upper_1d_aligned[i]:
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
        # Confluence breakout above both 4h and 1d upper Donchian with volume confirmation
        if bullish_confluence and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Confluence breakout below both 4h and 1d lower Donchian with volume confirmation
        elif bearish_confluence and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals