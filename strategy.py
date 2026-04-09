#!/usr/bin/env python3
# 1d_camarilla_1w_volume_v4
# Hypothesis: 1d strategy using weekly Camarilla pivot levels with volume confirmation.
# Enters long when price breaks above weekly H3 level with volume spike (>1.8x 20-day avg),
# short when price breaks below weekly L3 level with volume spike.
# Uses discrete position sizing (±0.30) to balance return and risk.
# Target: 50-120 total trades over 4 years (12-30/year). Works in bull/bear via pivot levels as dynamic S/R.
# Weekly HTF ensures alignment with major market structure, reducing whipsaw.
# Increased volume threshold and position size vs v3 to improve signal quality while maintaining trade frequency.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_1w_volume_v4"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels for weekly (H3/L3 for entry)
    h3_1w = pivot_1w + (range_1w * 1.1 / 4)
    l3_1w = pivot_1w - (range_1w * 1.1 / 4)
    
    # Align Camarilla levels to daily timeframe (completed weekly candle only)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    
    # Volume spike detection (20-day volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.8)  # Volume at least 1.8x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below weekly L3 level
            if close[i] < l3_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price rises above weekly H3 level
            if close[i] > h3_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Enter long: price breaks above weekly H3 with volume spike
            if (close[i] > h3_1w_aligned[i]) and vol_spike[i]:
                position = 1
                signals[i] = 0.30
            # Enter short: price breaks below weekly L3 with volume spike
            elif (close[i] < l3_1w_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.30
    
    return signals