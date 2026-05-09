#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Power with 1d trend filter and volume confirmation
# Long when Bull Power > 0, Bear Power < 0, 1d EMA50 uptrend, and volume > 1.5x average
# Short when Bear Power < 0, Bull Power > 0, 1d EMA50 downtrend, and volume > 1.5x average
# Exit when Elder Power signals reverse or volume drops below average
# Elder Ray measures bull/bear power relative to EMA13, providing dynamic support/resistance
# Designed to capture momentum shifts in both trending and ranging markets with controlled frequency
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_ElderRay_Power_1dEMA50_Volume"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Elder Ray (Bull Power and Bear Power) using EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, 1d EMA50 uptrend, volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0, Bull Power > 0, 1d EMA50 downtrend, volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] > 0 and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 or Bear Power >= 0 or volume drops
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 or Bull Power <= 0 or volume drops
            if (bear_power[i] >= 0 or bull_power[i] <= 0 or not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals