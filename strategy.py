#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for trend filter (1d)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to daily timeframe (1d primary)
    bull_power = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema13 = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Volume filter: volume > 1.5x 20-period average (1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema13[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Bull Power > 0 with price above EMA13 and volume spike
        # 2. Bear Power < 0 turning up with price above EMA13 and volume spike (early reversal)
        long_bull = (bull_power[i] > 0 and close[i] > ema13[i] and volume_spike[i])
        long_reversal = (bear_power[i] < 0 and bear_power[i] > bear_power[i-1] and 
                         close[i] > ema13[i] and volume_spike[i])
        
        # Short conditions:
        # 1. Bear Power < 0 with price below EMA13 and volume spike
        # 2. Bull Power > 0 turning down with price below EMA13 and volume spike (early reversal)
        short_bear = (bear_power[i] < 0 and close[i] < ema13[i] and volume_spike[i])
        short_reversal = (bull_power[i] > 0 and bull_power[i] < bull_power[i-1] and 
                          close[i] < ema13[i] and volume_spike[i])
        
        if long_bull or long_reversal:
            signals[i] = 0.25
            position = 1
        elif short_bear or short_reversal:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite power signal with volume confirmation
        elif position == 1 and bear_power[i] < 0 and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and bull_power[i] > 0 and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_ElderRay_BullBearPower_EMA13_Volume1.5x_1d"
timeframe = "1d"
leverage = 1.0