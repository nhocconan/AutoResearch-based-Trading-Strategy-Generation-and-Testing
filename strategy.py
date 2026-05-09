#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume spike
# Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and rising, EMA50 uptrend, volume > 1.5x average
# Short when Bear Power < 0 and falling, EMA50 downtrend, volume > 1.5x average
# Exit when power crosses zero or reverses
# Uses Elder Ray to measure bull/bear strength relative to EMA, EMA for trend, volume for conviction
# Designed to work in both trending and ranging markets with controlled frequency
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_ElderRay_EMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate 13-period EMA for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0 and rising, EMA50 uptrend, volume spike
            if (bull_power[i] > 0 and 
                bull_power[i] > bull_power[i-1] and  # Bull Power rising
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA50 rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 and falling, EMA50 downtrend, volume spike
            elif (bear_power[i] < 0 and 
                  bear_power[i] < bear_power[i-1] and  # Bear Power falling
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA50 falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power crosses below zero or reverses
            if (bull_power[i] <= 0) or (bull_power[i] < bull_power[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power crosses above zero or reverses
            if (bear_power[i] >= 0) or (bear_power[i] > bear_power[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals