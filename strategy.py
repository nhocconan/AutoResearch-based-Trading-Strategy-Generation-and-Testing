#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 Breakout + 1d Volume Spike + 1d Chop Regime Filter
# Long when: price breaks above Camarilla R1 (1d) AND 1d volume > 2x 20-period MA AND 1d Choppiness Index > 61.8 (range)
# Short when: price breaks below Camarilla S1 (1d) AND 1d volume > 2x 20-period MA AND 1d Choppiness Index > 61.8 (range)
# Exit when: price reverts to Camarilla Pivot Point (1d) OR Choppiness Index < 38.2 (trend)
# Uses Camarilla levels for mean reversion structure, volume for conviction, chop regime to avoid trending markets
# Timeframe: 4h, HTF: 1d for Camarilla, volume, and chop. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Camarilla_R1S1_Breakout_1dVolumeChop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for Camarilla, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels (based on previous day's range)
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We use previous day's values to avoid look-ahead
    pivot_1d = np.full(len(df_1d), np.nan)
    r1_1d = np.full(len(df_1d), np.nan)
    s1_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        hlc = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        rng = high_1d[i-1] - low_1d[i-1]
        pivot_1d[i] = hlc
        r1_1d[i] = close_1d[i-1] + (rng * 1.1 / 12.0)
        s1_1d[i] = close_1d[i-1] - (rng * 1.1 / 12.0)
    
    # Volume confirmation: 1d volume > 2x 20-period MA
    if len(volume_1d) >= 20:
        vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume_1d > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(len(df_1d), dtype=bool)
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR,14) / (ATR(14)*14)) / log10(14)
    # CHOP > 61.8 = range, CHOP < 38.2 = trend
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # ATR(14) using Wilder's smoothing
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        atr = wilder_smooth(tr, 14)
        tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        chop = 100 * np.log10(tr_sum_14 / (atr * 14)) / np.log10(14)
        chop_range = chop > 61.8   # range regime
        chop_trend = chop < 38.2   # trend regime (for exit)
    else:
        chop = np.full(len(df_1d), np.nan)
        chop_range = np.zeros(len(df_1d), dtype=bool)
        chop_trend = np.zeros(len(df_1d), dtype=bool)
    
    # Align all 1d indicators to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range.astype(float))
    chop_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_range_aligned[i]) or np.isnan(chop_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1 AND volume spike AND range regime
            if (close[i] > r1_1d_aligned[i] and 
                volume_spike_aligned[i] == 1.0 and 
                chop_range_aligned[i] == 1.0):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 AND volume spike AND range regime
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_spike_aligned[i] == 1.0 and 
                  chop_range_aligned[i] == 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to Pivot OR trend regime begins
            if (close[i] <= pivot_1d_aligned[i] or chop_trend_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to Pivot OR trend regime begins
            if (close[i] >= pivot_1d_aligned[i] or chop_trend_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals