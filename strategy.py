#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ElderRay_BullBearPower_ZeroCross_V1"
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
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on daily close (standard for Elder Ray)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13_1d
    
    # Align Bull and Bear Power to 6h timeframe (wait for daily close)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate volume spike (volume > 1.8 * 30-period average for moderate frequency)
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for EMA13
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation required
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when Bull Power crosses above zero (bullish momentum) with volume
            if bull_power_aligned[i] > 0 and bear_power_aligned[i-1] <= 0 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power crosses below zero (bearish momentum) with volume
            elif bear_power_aligned[i] < 0 and bull_power_aligned[i-1] >= 0 and vol_confirm:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when Bear Power crosses below zero (momentum shift)
            if bear_power_aligned[i] < 0 and bull_power_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when Bull Power crosses above zero (momentum shift)
            if bull_power_aligned[i] > 0 and bear_power_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals