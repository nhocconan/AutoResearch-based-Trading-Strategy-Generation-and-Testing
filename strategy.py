#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA13 for trend filter (fast EMA)
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_6h = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Elder Ray components on daily data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bull Power = High - EMA13
    bull_power_1d = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power_1d = low_1d - ema13_1d
    
    # Smooth the power values with EMA (13-period)
    bull_power_smooth_1d = pd.Series(bull_power_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power_smooth_1d = pd.Series(bear_power_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align smoothed power values to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_smooth_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_smooth_1d)
    
    # Volume spike detection (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(ema13_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish bias) with uptrend and volume spike
            if (bull_power_6h[i] > 0 and bear_power_6h[i] < 0 and 
                close[i] > ema13_6h[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 (bearish bias) with downtrend and volume spike
            elif (bear_power_6h[i] > 0 and bull_power_6h[i] < 0 and 
                  close[i] < ema13_6h[i] and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power turns positive OR trend turns down
            if bear_power_6h[i] > 0 or close[i] < ema13_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power turns positive OR trend turns up
            if bull_power_6h[i] > 0 or close[i] > ema13_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals