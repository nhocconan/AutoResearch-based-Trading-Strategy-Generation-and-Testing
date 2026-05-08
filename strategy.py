#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Ehlers Fisher Transform with 1d Trend Filter and Volume Confirmation
# - Fisher Transform identifies turning points in price
# - Long when Fisher crosses above -1.5 with 1d uptrend
# - Short when Fisher crosses below +1.5 with 1d downtrend
# - Volume filter ensures breakouts have momentum
# - Works in bull/bear by using 1d trend to avoid counter-trend trades
# - Target: 20-40 trades/year to minimize fee drag on 4h timeframe

name = "4h_FisherTransform_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Fisher Transform on 4h close
    price = close
    # Normalize price to [-1, 1] range over 10 periods
    highest_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    range_hl = highest_high - lowest_low
    # Avoid division by zero
    value1 = np.where(range_hl != 0, 2 * ((price - lowest_low) / range_hl - 0.5), 0)
    # Limit value1 to [-0.999, 0.999] to prevent log domain errors
    value1 = np.clip(value1, -0.999, 0.999)
    
    # Initialize Fisher arrays
    fish = np.full(n, np.nan)
    fish_signal = np.full(n, np.nan)
    
    # Calculate Fisher Transform
    for i in range(1, n):
        if i < 10:  # Need at least 10 periods for calculation
            continue
        value2 = 0.33 * value1[i] + 0.67 * fish[i-1] if not np.isnan(fish[i-1]) else 0.33 * value1[i]
        value2 = np.clip(value2, -0.999, 0.999)
        if np.isnan(fish[i-1]):
            fish[i] = 0.5 * np.log((1 + value2) / (1 - value2))
        else:
            fish[i] = 0.5 * np.log((1 + value2) / (1 - value2)) + 0.5 * fish[i-1]
        fish_signal[i] = fish[i-1]
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(fish[i]) or 
            np.isnan(fish_signal[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 with 1d uptrend + volume spike
            long_cond = (fish[i] > -1.5 and fish_signal[i] <= -1.5 and 
                        ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: Fisher crosses below +1.5 with 1d downtrend + volume spike
            short_cond = (fish[i] < 1.5 and fish_signal[i] >= 1.5 and 
                         ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Fisher crosses below signal line
            if fish[i] < fish_signal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Fisher crosses above signal line
            if fish[i] > fish_signal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals