#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d EMA50 trend filter and volume confirmation
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 AND volume > 1.5x average
# Short when Bear Power < 0 AND Bull Power > 0 AND price < 1d EMA50 AND volume > 1.5x average
# Exit when power signals weaken or price crosses EMA50 in opposite direction
# Elder Ray measures bull/bear power relative to EMA, effective in both trending and ranging markets
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "6h_ElderRay_Power_EMA50_Volume"
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
    
    # Calculate 13-period EMA for Elder Ray (standard setting)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Power components
    bull_power = high - ema13  # Bull Power: High - EMA
    bear_power = low - ema13   # Bear Power: Low - EMA
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
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
            # Enter long: Bull Power positive, Bear Power negative, price above EMA50, volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema50_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power negative, Bull Power positive, price below EMA50, volume spike
            elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                  close[i] < ema50_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Power weakens or price crosses below EMA50
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Power weakens or price crosses above EMA50
            if (bear_power[i] >= 0 or bull_power[i] <= 0 or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals