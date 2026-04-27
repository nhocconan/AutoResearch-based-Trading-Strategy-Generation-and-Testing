#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla pivot levels on 1d timeframe provide strong support/resistance levels.
Breakouts at R4/S4 with 1d trend alignment and volume confirmation capture strong momentum moves
while avoiding false breakouts. Designed for low trade frequency (15-25/year) to minimize fee drag.
Works in both bull and bear markets by following the 1d trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1. Calculate Camarilla pivot levels from 1d data (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r4 = pivot + (range_1d * 1.1)
    s4 = pivot - (range_1d * 1.1)
    
    # 2. 1d trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 3. Volume confirmation: current volume > 2.0 * 24-period average (1d equivalent on 6h)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align 1d indicators to 6h timeframe
    r4_1d = align_htf_to_ltf(prices, df_1d, r4)
    s4_1d = align_htf_to_ltf(prices, df_1d, s4)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Camarilla (1d), EMA34 (34), volume avg (24)
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_1d[i]) or np.isnan(s4_1d[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r4_level = r4_1d[i]
        s4_level = s4_1d[i]
        ema34 = ema34_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend: price vs EMA34 (1d)
            uptrend = close_val > ema34
            downtrend = close_val < ema34
            
            if uptrend and vol_conf:
                # Long: break above R4 with volume
                if close_val > r4_level:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short: break below S4 with volume
                if close_val < s4_level:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price re-enters below R4 or trend reversal
            if close_val < r4_level:  # Re-enter below R4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters above S4 or trend reversal
            if close_val > s4_level:  # Re-enter above S4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0