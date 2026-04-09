#!/usr/bin/env python3
# 1d_camarilla_1w_volume_v1
# Hypothesis: Daily strategy using weekly Camarilla pivot levels with volume confirmation.
# Enters long when price breaks above H3 level with volume spike, short when breaks below L3 level.
# Uses weekly HTF for structural context and discrete sizing (±0.25) to minimize fee churn.
# Target: 30-100 trades over 4 years. Works in bull/bear by using Camarilla levels as dynamic S/R.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_1w_volume_v1"
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
    
    # Get 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels (H3/L3 for entries, H4/L4 for stops)
    h3 = pivot + (range_1w * 1.1 / 4)
    l3 = pivot - (range_1w * 1.1 / 4)
    h4 = pivot + (range_1w * 1.1 / 2)
    l4 = pivot - (range_1w * 1.1 / 2)
    
    # Align weekly Camarilla levels to daily timeframe (completed weekly candle only)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
    # Volume spike detection (20-period volume average on daily)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below L3 level (stoploss)
            if close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above H3 level (stoploss)
            if close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 level with volume spike
            if (close[i] > h3_aligned[i]) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 level with volume spike
            elif (close[i] < l3_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals