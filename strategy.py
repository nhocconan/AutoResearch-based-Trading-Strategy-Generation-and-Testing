#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeSpike
# Hypothesis: Uses Camarilla R3/S3 levels on 1d for reversal/breakout signals, filtered by 12h EMA50 trend and volume spikes.
# In trending markets, breakouts above R3 or below S3 with volume and 12h trend alignment yield high-probability moves.
# In ranging markets, the filter reduces false breakouts. The 12h EMA ensures alignment with higher-timeframe trend.
# Volume spike confirms institutional participation. Designed for low trade frequency to minimize fee drag.
# Target: 20-40 trades/year per symbol for optimal performance on 4h.

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    # We need daily data; since we're on 4h, we'll use the prior day's close, high, low
    # Simplified: use rolling window of 6 bars (approx 1 day of 4h data) to get prior day's OHLC
    # This is an approximation; for exact daily, we'd use 1d data but we avoid resampling per rules
    # Instead, we calculate pivots on 1d data via mtf_data
    
    # Get 1d data for accurate Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # S3 = C - ((H-L)*1.1/4)
    # We'll use R3 and S3 as key levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid first day where shift is NaN
    valid_idx = ~np.isnan(prev_close)
    if not np.any(valid_idx):
        return np.zeros(n)
    
    rang = prev_high - prev_low
    R3 = prev_close + (rang * 1.1 / 4)
    S3 = prev_close - (rang * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 12h EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with volume and 12h uptrend
            if close[i] > R3_aligned[i] and volume_filter[i] and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and 12h downtrend
            elif close[i] < S3_aligned[i] and volume_filter[i] and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses back below R3 or 12h trend turns down
            if close[i] < R3_aligned[i] or ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses back above S3 or 12h trend turns up
            if close[i] > S3_aligned[i] or ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals