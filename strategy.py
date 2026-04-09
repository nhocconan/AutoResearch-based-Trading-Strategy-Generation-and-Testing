#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter
# Uses 1d Camarilla levels (H3/L3) for breakout entries in trending markets (CHOP < 38.2)
# Uses 1d Camarilla H4/L4 for mean reversion exits in ranging markets (CHOP > 61.8)
# Volume confirmation: 12h volume > 1.5 * 20-period average volume
# Discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: breakout catches trends, chop filter avoids whipsaws in ranging markets

name = "12h_1d_camarilla_breakout_volume_chop_v1"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels from prior day to avoid look-ahead
    # Camarilla: based on previous day's range
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels
    h3_1d = prev_close_1d + range_1d * 1.1 / 4
    l3_1d = prev_close_1d - range_1d * 1.1 / 4
    h4_1d = prev_close_1d + range_1d * 1.1 / 2
    l4_1d = prev_close_1d - range_1d * 1.1 / 2
    
    # Calculate 1d Choppiness Index (CHOP)
    def true_range(high, low, close):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        return tr
    
    atr_1d = true_range(high_1d, low_1d, close_1d)
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_smoothed_1d = wilders_smoothing(atr_1d, 14)
    
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_smoothed_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_1d - ll_1d
    chop_1d = np.where(range_14 != 0, 
                       100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 
                       50)
    
    # Align 1d indicators to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: 12h volume > 1.5 * 20-period average volume
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long: price below L4 (mean reversion) or ranging regime
            if close[i] < l4_1d_aligned[i] or chop_1d_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short: price above H4 (mean reversion) or ranging regime
            if close[i] > h4_1d_aligned[i] or chop_1d_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above H3 with volume confirmation in trending regime
            if (close[i] > h3_1d_aligned[i] and volume_confirmed[i] and 
                chop_1d_aligned[i] < 38.2):
                position = 1
                signals[i] = 0.25
            # Enter short: price below L3 with volume confirmation in trending regime
            elif (close[i] < l3_1d_aligned[i] and volume_confirmed[i] and 
                  chop_1d_aligned[i] < 38.2):
                position = -1
                signals[i] = -0.25
    
    return signals