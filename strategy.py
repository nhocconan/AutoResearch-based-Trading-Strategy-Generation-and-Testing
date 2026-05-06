#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d trend filter and volume confirmation
# Donchian breakouts capture momentum in both bull and bear markets
# 1d EMA50 filter ensures trades align with higher timeframe trend
# Volume confirmation (>1.5x 20-period average) filters false breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Target: 60-150 total trades over 4 years (15-37/year) with 0.20 position sizing

name = "1h_Donchian20_1dEMA50_VolumeSession_v1"
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
    
    # Calculate 4h Donchian channels ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 20-period Donchian channels on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate rolling max/min with period=20
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe
    donchian_high = align_htf_to_ltf(prices, df_4h, high_max_20)
    donchian_low = align_htf_to_ltf(prices, df_4h, low_min_20)
    
    # Calculate 1d EMA50 for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high with volume and above 1d EMA50
            if close[i] > donchian_high[i] and volume_filter[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low with volume and below 1d EMA50
            elif close[i] < donchian_low[i] and volume_filter[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals