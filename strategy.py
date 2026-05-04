#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla R3/S3 levels represent stronger support/resistance than R1/S1, reducing false breakouts.
# 4h EMA50 provides higher-timeframe trend bias to avoid counter-trend trades in both bull and bear markets.
# Volume spike (>2.0 x 20-period EMA) ensures institutional participation and reduces whipsaws.
# Designed for 1h timeframe targeting 60-150 total trades over 4 years (15-37/year).
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla pivots (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # R3 and S3 levels (more extreme than R1/S1)
    r3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i]) or np.isnan(pp_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # 4h trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_4h_aligned[i]
        bearish_trend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: Close breaks above R3 + volume spike + bullish 4h trend
            if (close[i] > r3_aligned[i] and volume_spike and bullish_trend):
                signals[i] = 0.20
                position = 1
            # Short: Close breaks below S3 + volume spike + bearish 4h trend
            elif (close[i] < s3_aligned[i] and volume_spike and bearish_trend):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Close drops below PP (pivot point) OR 4h trend turns bearish
            if (close[i] < pp_aligned[i] or bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Close rises above PP (pivot point) OR 4h trend turns bullish
            if (close[i] > pp_aligned[i] or bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals