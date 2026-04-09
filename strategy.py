#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and chop regime filter
# Camarilla pivots from 1d identify key intraday support/resistance levels
# Long when price touches L3 level with volume confirmation in low chop regime
# Short when price touches H3 level with volume confirmation in low chop regime
# Exit when price moves to opposite H3/L3 level or closes beyond H4/L4
# Uses discrete position sizing (0.25) to minimize fee churn
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot point = (high + low + close) / 3
    # Range = high - low
    # H4 = close + range * 1.1/2
    # H3 = close + range * 1.1/4
    # L3 = close - range * 1.1/4
    # L4 = close - range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h4_1d = close_1d + range_1d * 1.1 / 2.0
    h3_1d = close_1d + range_1d * 1.1 / 4.0
    l3_1d = close_1d - range_1d * 1.1 / 4.0
    l4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute choppiness index regime filter (14-period)
    # Choppy market: CHOP > 61.8 (range-bound, mean revert)
    # Trending market: CHOP < 38.2 (trend follow)
    # We want low chop regime for breakout trading: CHOP < 38.2
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=1).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=1).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=1).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # Avoid division by zero
    chop = 100 * np.log10(atr_14 * np.sqrt(14) / chop_denom) / np.log10(10)
    low_chop_regime = chop < 38.2  # Trending regime
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or
            np.isnan(l3_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or atr_14[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if not volume_confirmed or not low_chop_regime[i]:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when price reaches opposite H3 level or closes beyond H4
            if close[i] >= h3_1d_aligned[i] or close[i] > h4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when price reaches opposite L3 level or closes below L4
            if close[i] <= l3_1d_aligned[i] or close[i] < l4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Enter long when price touches L3 level with volume confirmation
            if abs(close[i] - l3_1d_aligned[i]) < (h4_1d_aligned[i] - l4_1d_aligned[i]) * 0.005:  # Within 0.5% of range
                position = 1
                signals[i] = position_size
            # Enter short when price touches H3 level with volume confirmation
            elif abs(close[i] - h3_1d_aligned[i]) < (h4_1d_aligned[i] - l4_1d_aligned[i]) * 0.005:  # Within 0.5% of range
                position = -1
                signals[i] = -position_size
    
    return signals