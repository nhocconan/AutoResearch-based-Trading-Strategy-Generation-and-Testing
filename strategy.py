#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
Long when price breaks above 20-day Donchian high with weekly EMA50 uptrend and volume > 1.5x average.
Short when price breaks below 20-day Donchian low with weekly EMA50 downtrend and volume > 1.5x average.
Exit when price returns to 10-day EMA (opposite direction).
Designed for low trade frequency (<15/year) to avoid fee drag while capturing major trends.
Works in both bull and bear markets via trend filter that adapts to weekly direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian channels and EMA - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_trend = ema_50_1w  # Rising when close > EMA50, falling when close < EMA50
    
    # Align daily indicators to 1d timeframe (already aligned, but use for safety)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10)
    
    # Align weekly trend to 1d timeframe (wait for weekly close)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Volume filter: 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_10_aligned[i]) or np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout + weekly uptrend + volume confirmation
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > weekly_trend_aligned[i] and  # Weekly uptrend: price above weekly EMA50
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown + weekly downtrend + volume confirmation
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < weekly_trend_aligned[i] and  # Weekly downtrend: price below weekly EMA50
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to 10-day EMA
                if close[i] <= ema_10_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to 10-day EMA
                if close[i] >= ema_10_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_WeeklyEMA50_Trend_VolumeFilter"
timeframe = "1d"
leverage = 1.0