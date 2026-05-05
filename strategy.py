#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Long when price breaks above 4h Camarilla R3 AND price > 1d EMA34 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 4h Camarilla S3 AND price < 1d EMA34 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses 4h Camarilla midpoint (R3/S3 midpoint) OR volume < avg_volume(20)
# Uses discrete sizing 0.20 to minimize fee churn
# Target: 60-150 total trades over 4 years (15-37/year)
# 4h Camarilla provides robust support/resistance; 1d EMA34 filters primary trend; volume confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
# Session filter 08-20 UTC to reduce noise

name = "1h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = prices.index.hour
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R3, S3, midpoint)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formula: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3_4h = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_s3_4h = close_4h - 1.1 * (high_4h - low_4h) / 2
    camarilla_mid_4h = (camarilla_r3_4h + camarilla_s3_4h) / 2
    
    # Align to 1h timeframe (completed 4h bars only)
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    camarilla_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_mid_4h)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_4h_aligned[i]) or np.isnan(camarilla_s3_4h_aligned[i]) or 
            np.isnan(camarilla_mid_4h_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, above 1d EMA34, volume confirmation
            if close[i] > camarilla_r3_4h_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3, below 1d EMA34, volume confirmation
            elif close[i] < camarilla_s3_4h_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla midpoint OR volume drops below average
            if close[i] < camarilla_mid_4h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price crosses above Camarilla midpoint OR volume drops below average
            if close[i] > camarilla_mid_4h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals