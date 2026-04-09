#!/usr/bin/env python3
# 1d_camarilla_1w_volume_atr_v1
# Hypothesis: 1d strategy using weekly Camarilla pivot levels for trend filter and daily Camarilla levels for entries.
# Enters long when price breaks above daily H3 with volume spike and price > weekly H4 (uptrend).
# Enters short when price breaks below daily L3 with volume spike and price < weekly L4 (downtrend).
# Uses ATR filter to ensure sufficient volatility. Discrete sizing (±0.25) to minimize fee churn.
# Target: 30-100 total trades over 4 years (7-25/year). Works in bull/bear via weekly trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_1w_volume_atr_v1"
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
    
    # Get 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels for 1d
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    h4_1d = pivot_1d + (range_1d * 1.1 / 2)
    l4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Calculate Camarilla pivot levels for 1w (trend filter)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    h4_1w = pivot_1w + (range_1w * 1.1 / 2)  # Weekly H4
    l4_1w = pivot_1w - (range_1w * 1.1 / 2)  # Weekly L4
    
    # Align Camarilla levels to 1d timeframe (completed HTF candle only)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h4_1w_aligned = align_htf_to_ltf(prices, df_1w, h4_1w)
    l4_1w_aligned = align_htf_to_ltf(prices, df_1w, l4_1w)
    
    # Volume spike detection (20-period volume average on 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    # ATR filter for volatility (14-period ATR on 1d)
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
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(h4_1w_aligned[i]) or np.isnan(l4_1w_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below daily L3 level
            if close[i] < l3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above daily H3 level
            if close[i] > h3_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above daily H3 with volume spike, ATR filter, and weekly uptrend (price > weekly H4)
            if (close[i] > h3_1d_aligned[i]) and vol_spike[i] and atr_filter[i] and (close[i] > h4_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below daily L3 with volume spike, ATR filter, and weekly downtrend (price < weekly L4)
            elif (close[i] < l3_1d_aligned[i]) and vol_spike[i] and atr_filter[i] and (close[i] < l4_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals