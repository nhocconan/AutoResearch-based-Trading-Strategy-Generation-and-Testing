#!/usr/bin/env python3
# mtf_1h_camarilla_4h1d_volume_v1
# Hypothesis: 1h strategy using 4h/1d Camarilla pivot levels with volume confirmation.
# Enters long when price breaks above H3 level (4h or 1d) with volume spike (>1.5x 20-bar avg).
# Enters short when price breaks below L3 level with volume spike.
# Uses discrete sizing (±0.20) to minimize fee churn. Target: 60-150 total trades over 4 years.
# Works in bull/bear by using HTF pivot levels as dynamic support/resistance and volume for confirmation.
# 1h timeframe for precise entry timing, 4h/1d for signal direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_camarilla_4h1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels for 4h
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    h3_4h = pivot_4h + (range_4h * 1.1 / 4)
    l3_4h = pivot_4h - (range_4h * 1.1 / 4)
    
    # Align 4h Camarilla levels to 1h timeframe (completed 4h candle only)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    
    # 1d HTF data for additional Camarilla confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    
    # Align 1d Camarilla levels to 1h timeframe (completed 1d candle only)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not in trading session or any required data is NaN
        if not in_session[i] or \
           (np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below L3 level (4h or 1d)
            if close[i] < l3_4h_aligned[i] or close[i] < l3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price rises above H3 level (4h or 1d)
            if close[i] > h3_4h_aligned[i] or close[i] > h3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price breaks above H3 level (4h OR 1d) with volume spike
            if ((close[i] > h3_4h_aligned[i] or close[i] > h3_1d_aligned[i]) and
                vol_spike[i]):
                position = 1
                signals[i] = 0.20
            # Enter short: price breaks below L3 level (4h OR 1d) with volume spike
            elif ((close[i] < l3_4h_aligned[i] or close[i] < l3_1d_aligned[i]) and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.20
    
    return signals