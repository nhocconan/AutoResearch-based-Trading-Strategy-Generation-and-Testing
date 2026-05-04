#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation
# Camarilla R3/S3 levels represent stronger support/resistance than R1/S1, reducing false breakouts.
# 1w EMA34 provides higher-timeframe trend bias to avoid counter-trend trades in both bull and bear markets.
# Volume spike (>2.0 x 20-period EMA) ensures institutional participation and reduces whipsaws.
# Designed for 1d timeframe targeting 30-100 total trades over 4 years (7-25/year).
# Works in bull markets via trend-aligned breakouts and in bear markets via filtered mean-reversion at extreme levels.

name = "1d_Camarilla_R3_S3_Breakout_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d_arr) / 3.0
    # R3 and S3 levels (more extreme than R1/S1)
    r3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 1d timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i]) or np.isnan(pp_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation: current volume > 2.0 x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # 1w trend: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = close[i] > ema_34_1w_aligned[i]
        bearish_trend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: Close breaks above R3 + volume spike + bullish 1w trend
            if (close[i] > r3_aligned[i] and volume_spike and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + volume spike + bearish 1w trend
            elif (close[i] < s3_aligned[i] and volume_spike and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close drops below PP (pivot point) OR 1w trend turns bearish
            if (close[i] < pp_aligned[i] or bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close rises above PP (pivot point) OR 1w trend turns bullish
            if (close[i] > pp_aligned[i] or bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals