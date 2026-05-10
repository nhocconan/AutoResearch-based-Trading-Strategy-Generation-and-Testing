#!/usr/bin/env python3
# 6h_FisherTransform_1dTrend_Volume
# Hypothesis: Uses Ehlers Fisher Transform on 6h price to detect reversals. Enters long when Fisher crosses above -1.5 with 1-day uptrend and volume confirmation.
# Enters short when Fisher crosses below +1.5 with 1-day downtrend and volume confirmation.
# Exits when Fisher crosses back through zero.
# Fisher Transform is effective in ranging and trending markets, capturing turning points with less lag than oscillators.
# Combined with daily trend filter to avoid counter-trend trades and volume to confirm conviction.
# Targets 15-35 trades per year on 6h timeframe with position size 0.25.

name = "6h_FisherTransform_1dTrend_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Fisher Transform on 6h prices (period=10)
    # Normalize price to [-1, 1] range over lookback period
    hl2 = (high + low) / 2
    max_hl2 = pd.Series(hl2).rolling(window=10, min_periods=10).max().values
    min_hl2 = pd.Series(hl2).rolling(window=10, min_periods=10).min().values
    # Avoid division by zero
    range_hl2 = max_hl2 - min_hl2
    range_hl2 = np.where(range_hl2 == 0, 1e-10, range_hl2)
    value = 2 * ((hl2 - min_hl2) / range_hl2 - 0.5)
    # Clamp to [-0.999, 0.999] for math.log stability
    value = np.clip(value, -0.999, 0.999)
    
    # Fisher Transform formula: 0.5 * ln((1+value)/(1-value))
    fish = 0.5 * np.log((1 + value) / (1 - value))
    # Smooth with 3-period EMA
    fish = pd.Series(fish).ewm(span=3, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Warmup for Fisher (10 + 3 smoothing)
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(fish[i]) or np.isnan(fish[i-1]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirm = volume > (vol_ma * 1.3)
        
        if position == 0:
            # Long entry: Fisher crosses above -1.5 with uptrend and volume
            if (fish[i] > -1.5 and fish[i-1] <= -1.5 and
                price_above_ema and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Fisher crosses below +1.5 with downtrend and volume
            elif (fish[i] < 1.5 and fish[i-1] >= 1.5 and
                  price_below_ema and
                  volume_confirm[i]):
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