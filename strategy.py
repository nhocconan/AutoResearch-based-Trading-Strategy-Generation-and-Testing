#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_power_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA(13) for Elder Ray power
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Smooth Elder Ray with 8-period EMA for signal line
    bull_ema = pd.Series(bull_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    bear_ema = pd.Series(bear_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    bull_ema_aligned = align_htf_to_ltf(prices, df_1d, bull_ema)
    bear_ema_aligned = align_htf_to_ltf(prices, df_1d, bear_ema)
    
    # Volume filter: 20-period average on 6h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup for EMA13
        # Skip if not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(bull_ema_aligned[i]) or np.isnan(bear_ema_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions:
        # Long: Bull Power > 0 and Bull Power > Bear Power (bulls in control) + volume
        long_signal = (bull_power_aligned[i] > 0) and (bull_power_aligned[i] > bear_power_aligned[i]) and volume_ok[i]
        # Short: Bear Power < 0 and Bear Power < Bull Power (bears in control) + volume
        short_signal = (bear_power_aligned[i] < 0) and (bear_power_aligned[i] < bull_power_aligned[i]) and volume_ok[i]
        
        # Exit when power crosses zero or reverses
        exit_long = (bull_power_aligned[i] <= 0) or (bull_power_aligned[i] < bear_power_aligned[i])
        exit_short = (bear_power_aligned[i] >= 0) or (bear_power_aligned[i] > bull_power_aligned[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals