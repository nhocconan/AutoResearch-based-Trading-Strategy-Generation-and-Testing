#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_v1
# Hypothesis: 12-hour breakouts at Camarilla pivot levels (H3/L3) from daily timeframe with volume confirmation (>2x 20-bar average volume).
# Camarilla levels act as support/resistance; breaks signal momentum continuation.
# Volume filter reduces false breakouts. Works in bull (upward breaks) and bear (downward breaks) markets.
# Target: 12-37 trades per year per symbol (~48-148 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily close for Camarilla levels
    daily_close = df_1d['close'].values
    
    # Camarilla levels: H3/L3 = C ± (H-L)*1.1/2
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    camarilla_h3 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_l3 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below L3 level
            if close[i] <= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above H3 level
            if close[i] >= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 with volume confirmation
            if close[i] > camarilla_h3_aligned[i] and volume[i] > vol_ma_20[i] * 2.0:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and volume[i] > vol_ma_20[i] * 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals