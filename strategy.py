#!/usr/bin/env python3
# 1d_camarilla_1w_volume_atr_v1
# Hypothesis: 1d strategy using weekly Camarilla pivot levels with volume confirmation and ATR filter.
# Enters long when price breaks above H3 level with volume spike and ATR > 0, short when breaks below L3 level.
# Uses discrete sizing (±0.30) to minimize fee churn. Target: 30-100 trades over 4 years.
# Works in bull/bear by using weekly Camarilla levels as dynamic support/resistance from higher timeframe.
# ATR filter ensures volatility is sufficient for breakout validity, reducing false signals.

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
    
    # Get 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for 1w
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    h3 = pivot + (range_1w * 1.1 / 4)
    l3 = pivot - (range_1w * 1.1 / 4)
    h4 = pivot + (range_1w * 1.1 / 2)
    l4 = pivot - (range_1w * 1.1 / 2)
    
    # Align Camarilla levels to 1d timeframe (completed 1w candle only)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    
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
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below L3 level
            if close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price rises above H3 level
            if close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Enter long: price breaks above H3 level with volume spike and ATR filter
            if (close[i] > h3_aligned[i]) and vol_spike[i] and atr_filter[i]:
                position = 1
                signals[i] = 0.30
            # Enter short: price breaks below L3 level with volume spike and ATR filter
            elif (close[i] < l3_aligned[i]) and vol_spike[i] and atr_filter[i]:
                position = -1
                signals[i] = -0.30
    
    return signals