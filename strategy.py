#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly Donchian breakout with daily trend filter and volume confirmation
# Weekly Donchian channels capture major trend breakouts
# Daily EMA50 trend filter ensures trades align with intermediate trend
# Volume > 1.5x 20-period average confirms institutional participation
# Works in bull markets (breakouts) and bear markets (breakdowns)
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_WeeklyDonchian20_DailyTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channels (20-period high/low) ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian high and low (20-period)
    high_20w = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Calculate daily EMA50 for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50
    ema_50d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align daily EMA50 to 12h timeframe
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or 
            np.isnan(ema_50d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high with volume confirmation and above daily EMA50
            if close[i] > high_20w_aligned[i] and volume_filter[i] and close[i] > ema_50d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly Donchian low with volume confirmation and below daily EMA50
            elif close[i] < low_20w_aligned[i] and volume_filter[i] and close[i] < ema_50d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low or above weekly Donchian high (take profit)
            if close[i] < low_20w_aligned[i] or close[i] > high_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high or below weekly Donchian low (take profit)
            if close[i] > high_20w_aligned[i] or close[i] < low_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals