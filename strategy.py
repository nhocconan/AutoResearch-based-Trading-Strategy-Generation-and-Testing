#!/usr/bin/env python3
# 12h_camarilla_1d_trend_volume_v3
# Hypothesis: 12h Camarilla pivot levels from 1d HTF + volume confirmation + 1d EMA50 trend filter.
# Long when price breaks above H3 with bullish volume and trend; short when breaks below L3 with bearish volume and trend.
# Exits when price reverts to pivot point or trend reverses. Designed to capture institutional level reactions with lower trade frequency.
# Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_trend_volume_v3"
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
    
    # 1d HTF data for Camarilla pivots and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d Camarilla pivot levels (based on previous day's range)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels
    h4 = pivot_point + (range_1d * 1.1 / 2)
    h3 = pivot_point + (range_1d * 1.1 / 4)
    h2 = pivot_point + (range_1d * 1.1 / 6)
    h1 = pivot_point + (range_1d * 1.1 / 12)
    l1 = pivot_point - (range_1d * 1.1 / 12)
    l2 = pivot_point - (range_1d * 1.1 / 6)
    l3 = pivot_point - (range_1d * 1.1 / 4)
    l4 = pivot_point - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: current volume > 2.0x 20-period average (stricter)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below H3 OR trend turns bearish OR price reaches pivot point (mean reversion)
            if close[i] < h3_aligned[i] or close[i] < ema50_1d_aligned[i] or close[i] <= pivot_point[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above L3 OR trend turns bullish OR price reaches pivot point (mean reversion)
            if close[i] > l3_aligned[i] or close[i] > ema50_1d_aligned[i] or close[i] >= pivot_point[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation (stricter)
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above H3 with bullish trend
                if close[i] > h3_aligned[i] and close[i] > ema50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below L3 with bearish trend
                elif close[i] < l3_aligned[i] and close[i] < ema50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals