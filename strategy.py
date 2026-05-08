#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + EMA200 trend filter + volume confirmation.
# Long when Bull Power > 0, price > EMA200, and volume > 1.5x 20-period average.
# Short when Bear Power < 0, price < EMA200, and volume > 1.5x 20-period average.
# Exit when Bull/Bear Power crosses zero or price crosses EMA200.
# Uses 6h timeframe with 1d EMA200 and volume for higher timeframe context.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency.

name = "6h_ElderRay_EMA200_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for EMA200 and volume
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 200:
        return np.zeros(n)
    
    # EMA200 on daily close
    ema200_d = pd.Series(df_d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily volume filter: current volume > 1.5x 20-period average
    volume_d = df_d['volume'].values
    vol_ma20_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = volume_d > (1.5 * vol_ma20_d)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_d)
    
    # Align EMA200 to 6h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_d, ema200_d)
    
    # Elder Ray components on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 200)  # Sufficient warmup for EMA13 and EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema200_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, price > EMA200, volume filter
            long_cond = (bull_power[i] > 0) and (close[i] > ema200_aligned[i]) and volume_filter[i]
            # Short conditions: Bear Power < 0, price < EMA200, volume filter
            short_cond = (bear_power[i] < 0) and (close[i] < ema200_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power crosses zero OR price crosses below EMA200
            if (bull_power[i] <= 0) or (close[i] <= ema200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power crosses zero OR price crosses above EMA200
            if (bear_power[i] >= 0) or (close[i] >= ema200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals