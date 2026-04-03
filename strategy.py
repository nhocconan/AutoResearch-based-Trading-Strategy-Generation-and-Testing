#!/usr/bin/env python3
"""
Experiment #1919: 6h Williams %R + 1d Camarilla Pivot Reversal + Volume Spike
HYPOTHESIS: Williams %R identifies overbought/oversold conditions on 6h, while 1d Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout) provide institutional support/resistance. Strategy: Enter on 6h Williams %R extreme (<10 for long, >90 for short) only when price is near 1d Camarilla R3/S3 levels (mean reversion zone) with volume confirmation (>1.5x average). Exit when Williams %R returns to neutral range (40-60) or opposite Camarilla level is touched. Works in ranging markets by fading extremes at institutional levels. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1919_6h_williamsr_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.1/2
    # R3 = C + Range * 1.1/4
    # S3 = C - Range * 1.1/4
    # S4 = C - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align 1d levels to 6h timeframe (shifted by 1 for completed bars only)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 6h Indicators: Williams %R(14) and Volume MA(20) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    williams_r[14:] = ((highest_high[14:] - close[14:]) / (highest_high[14:] - lowest_low[14:])) * -100
    
    # Volume MA(20) for spike detection
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
    
    warmup = 20  # sufficient for Williams %R(14) and volume MA(20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if Williams %R returns to neutral range (40-60)
                if williams_r[i] >= -40 and williams_r[i] <= -60:
                    exit_signal = True
                # Exit if price touches S4 (strong downside break)
                elif price <= s4_1d_aligned[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if Williams %R returns to neutral range (40-60)
                if williams_r[i] >= -40 and williams_r[i] <= -60:
                    exit_signal = True
                # Exit if price touches R4 (strong upside break)
                elif price >= r4_1d_aligned[i]:
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
            # Long entry: Williams %R oversold (<-90) AND price near S3 (support for mean reversion)
            if williams_r[i] < -90 and price >= s3_1d_aligned[i] * 0.995 and price <= s3_1d_aligned[i] * 1.005:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: Williams %R overbought (>-10) AND price near R3 (resistance for mean reversion)
            elif williams_r[i] > -10 and price >= r3_1d_aligned[i] * 0.995 and price <= r3_1d_aligned[i] * 1.005:
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