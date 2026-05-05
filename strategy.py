#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 4h for signal direction (trend + Camarilla levels) and 1h only for entry timing
# Session filter: 08-20 UTC to avoid low-volume Asian session noise
# Discrete sizing 0.20 to minimize fee churn
# Target: 60-150 total trades over 4 years (15-37/year) - avoids fee drag
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_time = prices['open_time']
    hours = pd.DatetimeIndex(open_time).hour  # Pre-compute for session filter
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for HTF indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 4h Camarilla levels (R3, S3, midpoint)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_last = df_4h['close'].values
    
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_4h_last + 1.1 * (high_4h - low_4h) / 2
    camarilla_s3 = close_4h_last - 1.1 * (high_4h - low_4h) / 2
    camarilla_mid = close_4h_last  # Camarilla midpoint is close
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_4h, camarilla_mid)
    
    # Volume confirmation: volume > 2.0 * 20-period average volume (calculated on 1h)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any value is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, above 4h EMA50, volume confirmation
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3, below 4h EMA50, volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla midpoint OR volume drops below average
            if close[i] < camarilla_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price crosses above Camarilla midpoint OR volume drops below average
            if close[i] > camarilla_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals