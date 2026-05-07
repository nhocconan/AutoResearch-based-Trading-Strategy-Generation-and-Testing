#!/usr/bin/env python3
"""
1d_Donchian_Breakout_WeeklyTrend_Volume_Confirmation
Hypothesis: On daily timeframe, buy when price breaks above 20-day Donchian high with weekly uptrend (EMA10 > EMA30) and volume confirmation; sell when breaks below 20-day Donchian low with weekly downtrend (EMA10 < EMA30) and volume confirmation. Uses weekly EMA for trend filter to avoid whipsaws and volume spike for confirmation. Designed for low trade frequency (<20/year) to minimize fee drag while capturing major trends in both bull and bear markets.
"""
name = "1d_Donchian_Breakout_WeeklyTrend_Volume_Confirmation"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA10 and EMA30 for trend filter
    ema10_weekly = pd.Series(close_weekly).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema30_weekly = pd.Series(close_weekly).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Align weekly EMA to daily timeframe
    ema10_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema10_weekly)
    ema30_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema30_weekly)
    
    # Calculate 20-day Donchian channels on daily data
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume filter: current volume > 2.0 * 50-day average volume
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donchian_window)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema10_weekly_aligned[i]) or np.isnan(ema30_weekly_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + volume filter
            if (close[i] > donchian_high[i] and 
                ema10_weekly_aligned[i] > ema30_weekly_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + weekly downtrend + volume filter
            elif (close[i] < donchian_low[i] and 
                  ema10_weekly_aligned[i] < ema30_weekly_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Donchian level
            if position == 1:
                if close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals