#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate 13-period EMA on daily close (Elder Ray base)
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Smooth the power values with 6-period EMA for less noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Align smoothed power values to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_smooth)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_smooth)
    
    # Volume confirmation: 20-period average on 6h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bull power positive AND rising, with volume confirmation
        if i >= 1:
            bull_rising = bull_power_aligned[i] > bull_power_aligned[i-1]
            if bull_power_aligned[i] > 0 and bull_rising and vol_confirm:
                enter_long = True
        
        # Short: Bear power negative AND falling, with volume confirmation
        if i >= 1:
            bear_falling = bear_power_aligned[i] < bear_power_aligned[i-1]
            if bear_power_aligned[i] < 0 and bear_falling and vol_confirm:
                enter_short = True
        
        # Exit conditions: power crosses zero
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bull power crosses below zero
            exit_long = bull_power_aligned[i] <= 0
        elif position == -1:
            # Exit short if bear power crosses above zero
            exit_short = bear_power_aligned[i] >= 0
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals