#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high + 1-day close > 1-day EMA50 + volume > 1.5x average.
Short when price breaks below 20-period Donchian low + 1-day close < 1-day EMA50 + volume > 1.5x average.
Exit when price crosses the opposite Donchian band or 1-day trend reverses.
Designed for moderate trade frequency (~20-40/year) to balance signal quality and fee efficiency.
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
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(daily_ema50_aligned[i]) or np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        daily_close_val = None
        daily_ema50_val = None
        if i < len(daily_ema50_aligned):
            daily_close_val = df_1d['close'].values[-1] if len(df_1d) > 0 else np.nan
            daily_ema50_val = daily_ema50_aligned[i]
        else:
            daily_close_val = np.nan
            daily_ema50_val = np.nan
            
        if np.isnan(daily_close_val) or np.isnan(daily_ema50_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_val > daily_ema50_val
        daily_trend_down = daily_close_val < daily_ema50_val
        
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + daily uptrend + volume confirmation
            if (close[i] > donchian_high[i] and 
                daily_trend_up and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + daily downtrend + volume confirmation
            elif (close[i] < donchian_low[i] and 
                  daily_trend_down and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Donchian low or daily trend changes to down
                if close[i] < donchian_low[i] or not daily_trend_up:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Donchian high or daily trend changes to up
                if close[i] > donchian_high[i] or not daily_trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_1dEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0