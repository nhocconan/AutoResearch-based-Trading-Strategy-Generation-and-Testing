#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ehlers Fisher Transform with 1d trend filter and volume confirmation.
# Fisher Transform highlights turning points with Gaussian probability distribution.
# Long when Fisher crosses above -1.5 AND price > 1d EMA50 AND volume > 1.5x 20-period average.
# Short when Fisher crosses below +1.5 AND price < 1d EMA50 AND volume > 1.5x 20-period average.
# Exit when Fisher crosses back through zero.
# Designed for 6h timeframe with low trade frequency (target: 10-25/year) to avoid fee drag.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades.
# Volume filter ensures participation and avoids low-conviction moves.
name = "6h_FisherTransform_1dEMA50_VolumeFilter"
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
    
    # Ehlers Fisher Transform (9-period)
    price = (high + low) / 2
    # Normalize price to [-1, 1] range over 9 periods
    max_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    min_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    range_val = max_high - min_low
    # Avoid division by zero
    range_val = np.where(range_val == 0, 1e-10, range_val)
    value1 = 2 * ((price - min_low) / range_val - 0.5)
    value1 = np.clip(value1, -0.999, 0.999)
    
    # Smooth with 2-period EMA
    value2 = np.zeros_like(value1)
    value2[0] = value1[0]
    for i in range(1, len(value1)):
        value2[i] = 0.5 * value1[i] + 0.5 * value2[i-1]
    
    # Fisher Transform
    fish = np.zeros_like(value2)
    fish[0] = 0.0
    for i in range(1, len(value2)):
        fish[i] = 0.5 * np.log((1 + value2[i]) / (1 - value2[i])) + 0.5 * fish[i-1]
    
    # EMA50 for trend
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(fish[i]) or np.isnan(ema50[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Fisher crosses above -1.5, price > 1d EMA50, volume filter
            fish_cross_up = fish[i] > -1.5 and fish[i-1] <= -1.5
            long_cond = fish_cross_up and (close[i] > ema50_1d_aligned[i]) and volume_filter[i]
            # Short conditions: Fisher crosses below +1.5, price < 1d EMA50, volume filter
            fish_cross_down = fish[i] < 1.5 and fish[i-1] >= 1.5
            short_cond = fish_cross_down and (close[i] < ema50_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Fisher crosses below zero
            if fish[i] < 0 and fish[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Fisher crosses above zero
            if fish[i] > 0 and fish[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals