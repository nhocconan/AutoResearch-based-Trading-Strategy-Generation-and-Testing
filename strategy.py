#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend and level filtering (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Load daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly EMA50 for long-term trend
    close_1w = df_1w['close'].values
    ema_50w_series = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50w = ema_50w_series.values
    
    # Previous day's pivot points (Classic style)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r1 = pivot + range_
    s1 = pivot - range_
    r2 = pivot + 2 * range_
    s2 = pivot - 2 * range_
    r3 = pivot + 3 * range_
    s3 = pivot - 3 * range_
    
    # Align weekly EMA and daily pivot levels to 6h timeframe
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50w)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: 24-period average (4 days)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_50w_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R2 with volume AND above weekly EMA50 (uptrend)
            if (close[i] > r2_aligned[i] and volume[i] > 1.8 * vol_avg_24[i] and 
                close[i] > ema_50w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S2 with volume AND below weekly EMA50 (downtrend)
            elif (close[i] < s2_aligned[i] and volume[i] > 1.8 * vol_avg_24[i] and 
                  close[i] < ema_50w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite S1/R1 level
            if position == 1:
                # Exit long: Price closes below S1
                if close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above R1
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Pivot_R2_S2_Breakout_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0