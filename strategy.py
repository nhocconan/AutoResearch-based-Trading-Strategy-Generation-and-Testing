#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 Breakout with 4h EMA50 Trend Filter and Volume Spike + Session Filter (08-20 UTC)
# Long when price breaks above R3 (4h) AND close > 4h EMA50 (uptrend) AND volume spike AND within session
# Short when price breaks below S3 (4h) AND close < 4h EMA50 (downtrend) AND volume spike AND within session
# R3/S3 are Camarilla levels (PP ± 1.1*range/4) for balanced breakout frequency
# EMA50 on 4h provides trend filter to reduce whipsaw
# Volume spike requires 2.0x 24-bar MA for confirmation
# Session filter (08-20 UTC) avoids low-liquidity hours
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation)
# Timeframe: 1h (primary)

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar (HLC of completed 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_shifted = np.roll(close_4h, 1)
    high_4h_shifted = np.roll(high_4h, 1)
    low_4h_shifted = np.roll(low_4h, 1)
    
    # Calculate pivot point (PP) = (H+L+C)/3
    pp = (high_4h_shifted + low_4h_shifted + close_4h_shifted) / 3.0
    # Calculate range
    range_4h = high_4h_shifted - low_4h_shifted
    # Camarilla levels (R3/S3 = PP ± 1.1*range/4)
    r3 = pp + (range_4h * 1.1 / 4.0)  # R3 = PP + 1.1*range/4
    s3 = pp - (range_4h * 1.1 / 4.0)  # S3 = PP - 1.1*range/4
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Volume confirmation on 1h with threshold
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_spike = volume > (2.0 * vol_ma_24)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    # Session filter: 08-20 UTC (precompute hours)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND uptrend (price > EMA50) AND volume spike AND in session
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i] and 
                in_session[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND downtrend (price < EMA50) AND volume spike AND in session
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i] and 
                  in_session[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below R3 OR closes below EMA50
            if close[i] < r3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above S3 OR closes above EMA50
            if close[i] > s3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals