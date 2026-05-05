#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 12h EMA34 Trend Filter and Volume Spike
# Long when price breaks above R3 (1d) AND close > 12h EMA34 (uptrend) AND volume spike
# Short when price breaks below S3 (1d) AND close < 12h EMA34 (downtrend) AND volume spike
# Uses Camarilla levels (R3/S3 = strong support/resistance) for high-probability breaks,
# 12h EMA34 for trend filter (avoid counter-trend trades), volume spike for conviction.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation).
# Timeframe: 4h (primary timeframe as required).

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed daily bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.roll(close_1d, 1)
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    
    # Calculate pivot point (PP) = (H+L+C)/3
    pp = (high_1d_shifted + low_1d_shifted + close_1d_shifted) / 3.0
    # Calculate range
    range_1d = high_1d_shifted - low_1d_shifted
    # Camarilla levels (R3/S3 = strong levels)
    r3 = pp + (range_1d * 1.1 / 4.0)  # R3 = PP + 1.1*range/4
    s3 = pp - (range_1d * 1.1 / 4.0)  # S3 = PP - 1.1*range/4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Higher threshold for fewer trades
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND uptrend (price > 12h EMA34) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND downtrend (price < 12h EMA34) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below R3 OR closes below 12h EMA34
            if close[i] < r3_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above S3 OR closes above 12h EMA34
            if close[i] > s3_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals