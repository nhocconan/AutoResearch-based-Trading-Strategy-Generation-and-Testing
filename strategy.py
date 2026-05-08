#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_DailyBullBearPower_EnergyShift"
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
    
    # Daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA13 for Elder Ray calculation
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = ema_13_1d - low_1d
    
    # Energy Shift: Bull Power - Bear Power (positive = bullish, negative = bearish)
    energy_shift_1d = bull_power_1d - bear_power_1d
    
    # Align to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    energy_shift_1d_aligned = align_htf_to_ltf(prices, df_1d, energy_shift_1d)
    
    # 6-day EMA of Energy Shift for trend smoothing
    energy_shift_ema6 = pd.Series(energy_shift_1d_aligned).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Volume confirmation: current volume > 1.3x 24-period average (4 days of 6h bars)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.3 * vol_ma24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(energy_shift_ema6[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Energy Shift turning positive with volume confirmation
            long_cond = (energy_shift_ema6[i] > 0 and 
                        energy_shift_ema6[i] > energy_shift_ema6[i-1] and
                        volume_confirm[i])
            
            # Short: Energy Shift turning negative with volume confirmation
            short_cond = (energy_shift_ema6[i] < 0 and 
                         energy_shift_ema6[i] < energy_shift_ema6[i-1] and
                         volume_confirm[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Energy Shift turns negative
            if energy_shift_ema6[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Energy Shift turns positive
            if energy_shift_ema6[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals