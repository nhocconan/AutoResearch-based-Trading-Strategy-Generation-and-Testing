#!/usr/bin/env python3
name = "6h_ElderRay_BullBearPower_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Ray components on daily data
    ema_period = 13
    ema_13 = pd.Series(close_1d).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Daily trend: EMA34
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current volume vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bull power > 0, bear power < 0, price above EMA34, volume confirmation
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                close[i] > ema_34_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: bear power < 0, bull power > 0, price below EMA34, volume confirmation
            elif (bear_power_aligned[i] < 0 and 
                  bull_power_aligned[i] > 0 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bear power turns positive OR bull power turns negative
            if (bear_power_aligned[i] >= 0 or 
                bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bull power turns negative OR bear power turns positive
            if (bull_power_aligned[i] <= 0 or 
                bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals