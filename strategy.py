#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d Trend Filter and Volume Spike
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with 1d uptrend and volume spike
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, with 1d downtrend and volume spike
# Uses 13-period EMA for responsiveness, 1d EMA50 for trend filter, volume > 1.5x 20-period EMA for confirmation
# Designed for low-frequency trades (target 50-150 total) to minimize fee drift

name = "6h_ElderRay_1dEMA50_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Elder Ray components (Bull Power, Bear Power) using 13-period EMA
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate 1-period change for Elder Ray momentum
    bull_power_change = np.diff(bull_power, prepend=0)
    bear_power_change = np.diff(bear_power, prepend=0)
    
    # Volume spike (1.5x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 and Elder Ray have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(bull_power_change[i]) or 
            np.isnan(bear_power_change[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0 and rising, Bear Power < 0, with 1d uptrend and volume spike
            if (bull_power[i] > 0 and bull_power_change[i] > 0 and 
                bear_power[i] < 0 and close[i] > ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 and falling, Bull Power > 0, with 1d downtrend and volume spike
            elif (bear_power[i] < 0 and bear_power_change[i] < 0 and 
                  bull_power[i] > 0 and close[i] < ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or Bear Power >= 0 or trend fails
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or Bull Power <= 0 or trend fails
            if (bear_power[i] >= 0 or bull_power[i] <= 0 or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals