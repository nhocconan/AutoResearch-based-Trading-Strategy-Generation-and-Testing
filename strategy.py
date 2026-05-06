#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-week Donchian breakout with volume confirmation and 1-day EMA trend filter
# Long when price breaks above 1-week Donchian upper channel with volume > 1.5x 20-period average and price above 1-day EMA50
# Short when price breaks below 1-week Donchian lower channel with volume > 1.5x 20-period average and price below 1-day EMA50
# Uses weekly Donchian for key support/resistance, volume for confirmation, and daily EMA for trend filter
# Designed to capture major trend continuations in both bull and bear markets with low trade frequency
# Target: 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing

name = "4h_1wDonchian20_1dEMA50_Volume_v1"
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
    
    # Calculate 1-week Donchian Channel (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-period high and low for weekly Donchian channels
    high_20 = df_1w['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 4h timeframe
    upper_donchian = align_htf_to_ltf(prices, df_1w, high_20)
    lower_donchian = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Calculate 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian with volume confirmation and above daily EMA50
            if close[i] > upper_donchian[i] and volume_filter[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly Donchian with volume confirmation and below daily EMA50
            elif close[i] < lower_donchian[i] and volume_filter[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian lower channel (support break)
            if close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly Donchian upper channel (resistance break)
            if close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals