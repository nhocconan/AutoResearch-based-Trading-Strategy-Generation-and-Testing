#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Breakout with Daily Trend Filter and Volume Confirmation
# Long when price breaks above weekly R1 AND price > daily EMA50 (uptrend) AND volume spike
# Short when price breaks below weekly S1 AND price < daily EMA50 (downtrend) AND volume spike
# Weekly pivots provide strong structural levels; daily EMA50 filters for trend alignment
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume)
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag
# Timeframe: 6h (primary)

name = "6h_WeeklyPivot_R1S1_Breakout_DailyEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot points from previous completed weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift by 1 to use only completed weekly bar (look-ahead safety)
    high_1w_shifted = np.roll(high_1w, 1)
    low_1w_shifted = np.roll(low_1w, 1)
    close_1w_shifted = np.roll(close_1w, 1)
    
    # Weekly pivot point (PP) = (H+L+C)/3
    pp = (high_1w_shifted + low_1w_shifted + close_1w_shifted) / 3.0
    # Weekly range
    range_1w = high_1w_shifted - low_1w_shifted
    # Weekly R1 and S1 levels
    r1 = pp + range_1w  # R1 = PP + range
    s1 = pp - range_1w  # S1 = PP - range
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation on 6h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 AND daily uptrend (price > EMA50) AND volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 AND daily downtrend (price < EMA50) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly R1 OR closes below daily EMA50
            if close[i] < r1_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly S1 OR closes above daily EMA50
            if close[i] > s1_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals