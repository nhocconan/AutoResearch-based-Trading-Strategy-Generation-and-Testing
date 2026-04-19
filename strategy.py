# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 20-bar Donchian breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-bar high AND weekly trend bullish (price > weekly EMA50) AND volume > 1.2x 20-bar avg volume.
# Short when price breaks below 20-bar low AND weekly trend bearish (price < weekly EMA50) AND volume > 1.2x 20-bar avg volume.
# Exit when price crosses back below/above 20-bar average (mean of high and low).
# Uses Donchian for breakout structure, weekly EMA for trend filter, volume for confirmation.
# Target: 12-37 trades/year per symbol (50-150 total over 4 years).
name = "12h_Donchian_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian calculation (using daily to build 20-bar period)
    df_1d = get_htf_data(prices, '1d')
    
    # 20-bar Donchian channels on daily data
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Donchian middle line (exit condition)
    donchian_mid = (high_20 + low_20) / 2
    
    # Align Donchian channels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Get weekly data for trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    weekly_ema50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Get 20-bar average volume on daily data for confirmation
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure Donchian and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(weekly_ema50_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_20_val = high_20_aligned[i]
        low_20_val = low_20_aligned[i]
        donchian_mid_val = donchian_mid_aligned[i]
        weekly_ema = weekly_ema50_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        vol = volume[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.2 * vol_ma
        
        if position == 0:
            # Long entry: break above 20-bar high + weekly bullish trend + volume confirmation
            if price > high_20_val and price > weekly_ema and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: break below 20-bar low + weekly bearish trend + volume confirmation
            elif price < low_20_val and price < weekly_ema and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian middle line
            if price < donchian_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian middle line
            if price > donchian_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals