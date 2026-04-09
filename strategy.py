#!/usr/bin/env python3
# 4h_12h_camarilla_pivot_v1
# Hypothesis: 4h strategy using 12h Camarilla pivot levels with volume confirmation and ATR filter.
# Enters long when price breaks above 12h H3 level with volume spike and ATR > 0,
# short when price breaks below 12h L3 level with volume spike and ATR > 0.
# Uses discrete position sizing (±0.25) to minimize fee churn.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull/bear via pivot levels as dynamic S/R.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels for 12h
    h3_12h = pivot_12h + (range_12h * 1.1 / 4)
    l3_12h = pivot_12h - (range_12h * 1.1 / 4)
    h4_12h = pivot_12h + (range_12h * 1.1 / 2)
    l4_12h = pivot_12h - (range_12h * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (completed HTF candle only)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    
    # Volume spike detection (20-period volume average on 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    # ATR filter for volatility (14-period ATR on 4h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period: use only high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_filter = atr > 0  # Ensure ATR is valid (non-zero)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_12h_aligned[i]) or np.isnan(l3_12h_aligned[i]) or
            np.isnan(h4_12h_aligned[i]) or np.isnan(l4_12h_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below 12h L3 level
            if close[i] < l3_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above 12h H3 level
            if close[i] > h3_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 12h H3 with volume spike and ATR filter
            if (close[i] > h3_12h_aligned[i]) and vol_spike[i] and atr_filter[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 12h L3 with volume spike and ATR filter
            elif (close[i] < l3_12h_aligned[i]) and vol_spike[i] and atr_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals