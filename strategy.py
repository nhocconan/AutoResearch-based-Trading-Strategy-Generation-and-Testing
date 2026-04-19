#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day pivot levels (R1/S1, R2/S2) and volume confirmation.
# Buy near S1/S2 with bullish momentum, sell near R1/R2 with bearish momentum.
# Uses daily pivot points calculated from prior day's OHLC. Works in both bull and bear markets by
# buying dips to support in uptrends and selling rallies to resistance in downtrends.
# Target: 75-200 total trades over 4 years (19-50/year).
name = "4h_1d_Pivot_R1S1_R2S2_Volume_Momentum"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day's OHLC)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    s1_1d = (2 * pivot_1d) - high_1d
    r1_1d = (2 * pivot_1d) - low_1d
    s2_1d = pivot_1d - (high_1d - low_1d)
    r2_1d = pivot_1d + (high_1d - low_1d)
    
    # Align daily pivot levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    # Momentum filter: price > 10-period EMA for bullish, < 10-period EMA for bearish
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or
            np.isnan(s2_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(ema_10[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long near support with bullish momentum
            if close[i] <= s1_1d_aligned[i] * 1.005 and close[i] >= s1_1d_aligned[i] * 0.995 and volume_filter[i] and close[i] > ema_10[i]:
                signals[i] = 0.25
                position = 1
            # Long near S2 with bullish momentum
            elif close[i] <= s2_1d_aligned[i] * 1.005 and close[i] >= s2_1d_aligned[i] * 0.995 and volume_filter[i] and close[i] > ema_10[i]:
                signals[i] = 0.25
                position = 1
            # Short near resistance with bearish momentum
            elif close[i] >= r1_1d_aligned[i] * 0.995 and close[i] <= r1_1d_aligned[i] * 1.005 and volume_filter[i] and close[i] < ema_10[i]:
                signals[i] = -0.25
                position = -1
            # Short near R2 with bearish momentum
            elif close[i] >= r2_1d_aligned[i] * 0.995 and close[i] <= r2_1d_aligned[i] * 1.005 and volume_filter[i] and close[i] < ema_10[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit at R1 (take profit) or S2 (stop reversal)
            if close[i] >= r1_1d_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            elif close[i] <= s2_1d_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit at S1 (take profit) or R2 (stop reversal)
            if close[i] <= s1_1d_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            elif close[i] >= r2_1d_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals