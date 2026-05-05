#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout + Volume Spike + Chop Regime Filter
# Uses 1d Camarilla levels (R3/S3) for structure, 12h for entry timing with volume confirmation
# Chop filter (EHLERS) avoids whipsaws in ranging markets
# Discrete sizing 0.25 to minimize fee churn
# Works in bull (breakout continuation) and bear (breakdown continuation) markets
# Target: 80-120 total trades over 4 years (20-30/year)

name = "12h_Camarilla_R3S3_Breakout_Volume_Chop"
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
    
    # Get 1d data ONCE before loop for Camarilla levels and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (R3, S3)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    #            S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    daily_range = high_1d - low_1d
    camarilla_r3 = close_1d + 1.125 * daily_range
    camarilla_s3 = close_1d - 1.125 * daily_range
    
    # Align 1d Camarilla levels to 12h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate EHLERS Chop Index on 1d (regime filter)
    # Chop = 100 * log10(sum(ATR1)/sum(range)) / log10(n)
    # Chop > 61.8 = ranging, Chop < 38.2 = trending
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_1d = np.array([
        true_range(high_1d[i], low_1d[i], close_1d[i-1] if i > 0 else close_1d[0])
        for i in range(len(close_1d))
    ])
    
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    range_1d = high_1d - low_1d
    
    # Chop calculation over 14-period window
    chop_1d = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        sum_atr = np.sum(tr_1d[i-13:i+1])
        sum_range = np.sum(range_1d[i-13:i+1])
        if sum_range > 0:
            chop_1d[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(sum_range) * np.log10(sum_range)
        else:
            chop_1d[i] = 50  # neutral
    
    # Align Chop to 12h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume spike detection on 12h (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)  # 50% above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime: only trade when Chop < 50 (not strongly ranging)
        # Chop > 61.8 = strong range (avoid), Chop < 38.2 = strong trend
        # We allow Chop < 50 to avoid whipsaws but still catch trends
        if chop_1d_aligned[i] > 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 or volume dries up
            if close[i] < camarilla_r3_aligned[i] or volume[i] < volume_ma[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 or volume dries up
            if close[i] > camarilla_s3_aligned[i] or volume[i] < volume_ma[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals