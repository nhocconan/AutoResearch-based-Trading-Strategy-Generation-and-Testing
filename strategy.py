#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 12-hour trend filter and volume confirmation
# Go long when Bull Power > 0, Bear Power < 0, 12h EMA(50) uptrend, and volume spike
# Go short when Bull Power < 0, Bear Power > 0, 12h EMA(50) downtrend, and volume spike
# Uses Elder Ray to capture institutional buying/selling pressure combined with trend alignment
# Designed to work in both bull and bear markets by requiring 12h trend confirmation
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_ElderRay_BullBear_12hTrend_Volume"
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
    
    # Get 12h data once for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Elder Ray components (13-period EMA as standard)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA(13)
    bear_power = low - ema13   # Bear Power = Low - EMA(13)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_12h_val = ema50_12h_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, 12h uptrend, volume spike
            if (bull > 0 and bear < 0 and 
                close[i] > ema50_12h_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Bull Power < 0, Bear Power > 0, 12h downtrend, volume spike
            elif (bull < 0 and bear > 0 and 
                  close[i] < ema50_12h_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR 12h trend turns down
            if (bull <= 0 or bear >= 0 or close[i] < ema50_12h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power >= 0 OR Bear Power <= 0 OR 12h trend turns up
            if (bull >= 0 or bear <= 0 or close[i] > ema50_12h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals