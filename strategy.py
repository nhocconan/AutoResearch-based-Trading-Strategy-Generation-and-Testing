#!/usr/bin/env python3
# 6h_1d_camarilla_pivot_volume_v1
# Hypothesis: Camarilla pivot levels on 1d provide reliable support/resistance. Fade at R3/S3 levels with volume exhaustion signals, breakout continuation at R4/S4 with volume expansion. Works in both bull/bear markets by fading overextensions in ranging markets and capturing breakouts in trending markets. 6h timeframe reduces overtrading vs lower timeframes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day has no previous data
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    camarilla_h4 = prev_close + range_ * 1.1 / 2
    camarilla_l4 = prev_close - range_ * 1.1 / 2
    camarilla_h3 = prev_close + range_ * 1.1 / 4
    camarilla_l3 = prev_close - range_ * 1.1 / 4
    camarilla_h2 = prev_close + range_ * 1.1 / 6
    camarilla_l2 = prev_close - range_ * 1.1 / 6
    camarilla_h1 = prev_close + range_ * 1.1 / 12
    camarilla_l1 = prev_close - range_ * 1.1 / 12
    
    # Align Camarilla levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume exhaustion: current volume < 0.7x average of last 6 periods
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    vol_exhaustion = volume < vol_ma * 0.7
    
    # Volume expansion: current volume > 1.8x average of last 6 periods
    vol_expansion = volume > vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if Camarilla data not available
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3 or reaches L4 (target)
            if close[i] < l3_aligned[i] or close[i] <= l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 or reaches H4 (target)
            if close[i] > h3_aligned[i] or close[i] >= h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Fade at R3/S3 with volume exhaustion (mean reversion)
            if close[i] > h3_aligned[i] and vol_exhaustion[i]:
                position = -1  # Short at R3
                signals[i] = -0.25
            elif close[i] < l3_aligned[i] and vol_exhaustion[i]:
                position = 1   # Long at S3
                signals[i] = 0.25
            # Breakout continuation at R4/S4 with volume expansion
            elif close[i] > h4_aligned[i] and vol_expansion[i]:
                position = 1   # Long breakout
                signals[i] = 0.25
            elif close[i] < l4_aligned[i] and vol_expansion[i]:
                position = -1  # Short breakdown
                signals[i] = -0.25
    
    return signals