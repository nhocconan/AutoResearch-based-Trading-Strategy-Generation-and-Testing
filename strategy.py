#!/usr/bin/env python3
"""
Experiment #1899: 6h Donchian(20) Breakout + 12h Camarilla Pivot + Volume Spike
HYPOTHESIS: Donchian breakouts capture momentum. Camarilla pivots from 12h define key S/R levels (R3/S3 for mean reversion, R4/S4 for breakout). Combined with volume confirmation (>1.5x average) and requiring alignment between 6h breakout direction and 12h pivot bias, this strategy filters false breakouts. Works in both bull/bear by following institutional levels. Target: 75-150 total trades over 4 years (19-37/year) with discrete position sizing of 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1899_6h_donchian20_12h_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    # S4 = C - (H-L) * 1.1/2
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r4_12h = close_12h + range_12h * 1.1 / 2.0
    r3_12h = close_12h + range_12h * 1.1 / 4.0
    s3_12h = close_12h - range_12h * 1.1 / 4.0
    s4_12h = close_12h - range_12h * 1.1 / 2.0
    
    # Align Camarilla levels to 6h
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for Donchian(20) and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Reverse signal or adverse move ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below 6h Donchian lower band (20)
                if price < lowest_low[i]:
                    exit_signal = True
                # Exit if price reaches 12h S4 (strong support - take profit)
                elif price <= s4_12h_aligned[i]:
                    exit_signal = True
                # Exit if price reverses from 12h R3 (strong resistance)
                elif price >= r3_12h_aligned[i] and bars_since_entry >= 2:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above 6h Donchian upper band (20)
                if price > highest_high[i]:
                    exit_signal = True
                # Exit if price reaches 12h R4 (strong resistance - take profit)
                elif price >= r4_12h_aligned[i]:
                    exit_signal = True
                # Exit if price reverses from 12h S3 (strong support)
                elif price <= s3_12h_aligned[i] and bars_since_entry >= 2:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine 12h pivot bias
            # Bias long if price above R3, short if price below S3
            pivot_bias = 0
            if price > r3_12h_aligned[i]:
                pivot_bias = 1  # bullish bias
            elif price < s3_12h_aligned[i]:
                pivot_bias = -1  # bearish bias
            
            # Long entry: price breaks above 6h Donchian upper band (20) AND bullish pivot bias
            if pivot_bias > 0 and price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below 6h Donchian lower band (20) AND bearish pivot bias
            elif pivot_bias < 0 and price < lowest_low[i]:
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