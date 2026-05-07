#!/usr/bin/env python3
# 4h_1dDonchian_Breakout_1dTrend_Volume
# Uses daily Donchian channels for trend detection with volume confirmation.
# Long when price breaks above 20-day high in uptrend, short when breaks below 20-day low in downtrend.
# Designed for 4h timeframe to capture major trend moves with volume confirmation.

name = "4h_1dDonchian_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: highest high over last 20 days
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over last 20 days
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter (20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above Donchian high with uptrend and volume
            if close[i] > donchian_high_4h[i] and close[i] > ema_50_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below Donchian low with downtrend and volume
            elif close[i] < donchian_low_4h[i] and close[i] < ema_50_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price returns below EMA50 or breaks below Donchian low
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry >= 3 and (close[i] < ema_50_4h[i] or close[i] < donchian_low_4h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above EMA50 or breaks above Donchian high
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry >= 3 and (close[i] > ema_50_4h[i] or close[i] > donchian_high_4h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals