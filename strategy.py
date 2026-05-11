#!/usr/bin/env python3
name = "1d_WeeklyBreakout_TrendVolume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend and Donchian breakout
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Weekly high and low for Donchian breakout (using previous week's data)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly Donchian channels (20-period) - using previous week's data
    # Calculate 20-period high and low from weekly data
    high_series_w = pd.Series(high_w)
    low_series_w = pd.Series(low_w)
    donchian_high_w = high_series_w.rolling(window=20, min_periods=20).max().values
    donchian_low_w = low_series_w.rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (using previous week's completed data)
    donchian_high_d = align_htf_to_ltf(prices, df_w, donchian_high_w)
    donchian_low_d = align_htf_to_ltf(prices, df_w, donchian_low_w)
    
    # Weekly trend filter: EMA20 on weekly close
    close_series_w = pd.Series(close_w)
    ema_w = close_series_w.ewm(span=20, min_periods=20).mean().values
    ema_w_aligned = align_htf_to_ltf(prices, df_w, ema_w)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40  # Need enough data for weekly calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_d[i]) or np.isnan(donchian_low_d[i]) or 
            np.isnan(ema_w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high AND above weekly EMA20 (uptrend) AND volume surge
            if close[i] > donchian_high_d[i] and close[i] > ema_w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low AND below weekly EMA20 (downtrend) AND volume surge
            elif close[i] < donchian_low_d[i] and close[i] < ema_w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below weekly Donchian low OR below weekly EMA20 (trend change)
            if close[i] < donchian_low_d[i] or close[i] < ema_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above weekly Donchian high OR above weekly EMA20 (trend change)
            if close[i] > donchian_high_d[i] or close[i] > ema_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals