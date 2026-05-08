#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
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
    
    # Volume spike: current volume > 1.8x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma24)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA26 for trend filter (faster than EMA34 for more signals)
    ema_26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_26_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_26_1d)
    
    # Calculate Elder Ray components (13-period EMA for power)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 26  # Sufficient warmup for EMA26 and EMA13
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_26_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, price above 1d EMA26, volume spike
            long_cond = (bull_power[i] > 0) and (close[i] > ema_26_1d_aligned[i]) and volume_spike[i]
            # Short conditions: Bear Power < 0, price below 1d EMA26, volume spike
            short_cond = (bear_power[i] < 0) and (close[i] < ema_26_1d_aligned[i]) and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR price crosses below 1d EMA26
            if (bull_power[i] <= 0) or (close[i] < ema_26_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive OR price crosses above 1d EMA26
            if (bear_power[i] >= 0) or (close[i] > ema_26_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals