#!/usr/bin/env python3
"""
Hypothesis: 6-hour Fisher Transform with 1-day trend filter and volume confirmation.
Long when Fisher crosses above -1.5 (bullish reversal) + daily close > daily EMA50 + volume > 1.5x average.
Short when Fisher crosses below +1.5 (bearish reversal) + daily close < daily EMA50 + volume > 1.5x average.
Exit when Fisher crosses zero (momentum exhaustion).
Designed for low trade frequency (~15-30/year) to minimize fee drag in both bull and bear markets.
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
    
    # Calculate Fisher Transform (10-period)
    hl2 = (high + low) / 2
    max_hl2 = pd.Series(hl2).rolling(window=10, min_periods=10).max()
    min_hl2 = pd.Series(hl2).rolling(window=10, min_periods=10).min()
    range_hl2 = max_hl2 - min_hl2
    # Avoid division by zero
    range_hl2 = np.where(range_hl2 == 0, 1e-10, range_hl2)
    value = 2 * ((hl2 - min_hl2) / range_hl2 - 0.5)
    # Clamp value to [-0.999, 0.999] for arctanh stability
    value = np.clip(value, -0.999, 0.999)
    fish = 0.5 * np.log((1 + value) / (1 - value))  # atanh
    fish = pd.Series(fish).ewm(span=3, adjust=False).mean().values  # smooth
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        # Skip if data not ready
        if (np.isnan(fish[i]) or np.isnan(daily_ema50_aligned[i]) or 
            np.isnan(avg_volume[i]) or volume[i] == 0):
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
            # Long: Fisher crosses above -1.5 + daily uptrend + volume confirmation
            if (fish[i] > -1.5 and fish[i-1] <= -1.5 and 
                daily_trend_up and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below +1.5 + daily downtrend + volume confirmation
            elif (fish[i] < 1.5 and fish[i-1] >= 1.5 and 
                  daily_trend_down and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Fisher crosses above +1.5 or daily trend changes to down
                if fish[i] >= 1.5 or not daily_trend_up:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Fisher crosses below -1.5 or daily trend changes to up
                if fish[i] <= -1.5 or not daily_trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Fisher_DailyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0