#!/usr/bin/env python3
# 12h_camarilla_1w_volume_v1
# Hypothesis: 12h strategy using weekly Camarilla pivot levels with volume confirmation.
# Enters long when price breaks above weekly H3 level with volume spike, short when breaks below weekly L3 level.
# Uses weekly H4/L4 as trend confirmation: only take longs when price > weekly H4 (uptrend), shorts when price < weekly L4 (downtrend).
# ATR filter ensures sufficient volatility for breakout validity. Discrete sizing (±0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull/bear via Camarilla levels as dynamic S/R from higher TF.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1w_volume_v1"
timeframe = "12h"
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
    
    # Calculate Camarilla pivot levels for 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels for 1w
    h3_1w = pivot_1w + (range_1w * 1.1 / 4)
    l3_1w = pivot_1w - (range_1w * 1.1 / 4)
    h4_1w = pivot_1w + (range_1w * 1.1 / 2)  # Weekly H4
    l4_1w = pivot_1w - (range_1w * 1.1 / 2)  # Weekly L4
    
    # Align Camarilla levels to 12h timeframe (completed HTF candle only)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    h4_1w_aligned = align_htf_to_ltf(prices, df_1w, h4_1w)
    l4_1w_aligned = align_htf_to_ltf(prices, df_1w, l4_1w)
    
    # Volume spike detection (20-period volume average on 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    # ATR filter for volatility (14-period ATR on 12h)
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
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or
            np.isnan(h4_1w_aligned[i]) or np.isnan(l4_1w_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below weekly L3 level
            if close[i] < l3_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above weekly H3 level
            if close[i] > h3_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly H3 with volume spike, ATR filter, and price > weekly H4 (uptrend)
            if (close[i] > h3_1w_aligned[i]) and vol_spike[i] and atr_filter[i] and (close[i] > h4_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly L3 with volume spike, ATR filter, and price < weekly L4 (downtrend)
            elif (close[i] < l3_1w_aligned[i]) and vol_spike[i] and atr_filter[i] and (close[i] < l4_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals