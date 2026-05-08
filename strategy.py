#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA13 trend filter and volume confirmation
# Long when Bull Power > 0, Bear Power < 0, price > EMA13(1d), volume > 1.5x average
# Short when Bear Power < 0, Bull Power > 0, price < EMA13(1d), volume > 1.5x average
# Uses 13-period EMA for smoothing to reduce whipsaws in both bull and bear markets
# Targets 50-150 total trades over 4 years (12-37/year) for optimal balance of signal and cost

name = "6h_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for EMA13 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # We need EMA13 of close for each 6h bar to calculate Elder Ray
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13_6h  # Bull Power = High - EMA13
    bear_power = low - ema13_6h   # Bear Power = Low - EMA13
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # warmup for EMA13
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        ema13_1d_val = ema13_1d_aligned[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, price > EMA13(1d), volume spike
            if bull_power_val > 0 and bear_power_val < 0 and close_val > ema13_1d_val and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0, Bull Power > 0, price < EMA13(1d), volume spike
            elif bear_power_val < 0 and bull_power_val > 0 and close_val < ema13_1d_val and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power >= 0 or price < EMA13(1d)
            if bear_power_val >= 0 or close_val < ema13_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power <= 0 or price > EMA13(1d)
            if bull_power_val <= 0 or close_val > ema13_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals