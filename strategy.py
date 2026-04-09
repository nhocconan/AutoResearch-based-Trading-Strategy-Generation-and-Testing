#!/usr/bin/env python3
# 12h_camarilla_1d_trend_volume_v2
# Hypothesis: 12h strategy using daily Camarilla H3/L3 levels with 1d trend filter (price above/below daily EMA200) and volume confirmation (>1.5x 20-period average).
# In bull markets: long when price touches L3 support in uptrend (price > EMA200) with volume spike.
# In bear markets: short when price touches H3 resistance in downtrend (price < EMA200) with volume spike.
# Uses discrete position sizing (±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots and EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1d Camarilla pivot levels (based on previous day's range)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels: H3, L3 (strongest intraday support/resistance)
    h3 = pivot_point + (range_1d * 1.1 / 4)
    l3 = pivot_point - (range_1d * 1.1 / 4)
    
    # Align 1d data to 12h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR below daily EMA200 (trend fails)
            if close[i] < l3_aligned[i] or close[i] < ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR above daily EMA200 (trend fails)
            if close[i] > h3_aligned[i] or close[i] > ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price touches L3 support in uptrend (price > EMA200)
                if abs(close[i] - l3_aligned[i]) < (h3_aligned[i] - l3_aligned[i]) * 0.02 and close[i] > ema200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches H3 resistance in downtrend (price < EMA200)
                elif abs(close[i] - h3_aligned[i]) < (h3_aligned[i] - l3_aligned[i]) * 0.02 and close[i] < ema200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals