#!/usr/bin/env python3
# 6H_FisherTransform_1dTrend_VolumeSpike
# Hypothesis: Ehlers Fisher Transform on 6h chart for reversal signals, filtered by 1d trend (EMA50) and volume spikes.
# Long: Fisher crosses above -1.5 in uptrend (close > EMA50) with volume > 2.5x 20-period average.
# Short: Fisher crosses below +1.5 in downtrend (close < EMA50) with volume confirmation.
# Exit when Fisher crosses back through zero (mean reversion) or trend reverses.
# Uses 1d EMA50 for trend to avoid whipsaws, works in both bull/bear markets.
# Targets 12-37 trades per year on 6h timeframe with position size 0.25 to minimize fee drag.

name = "6H_FisherTransform_1dTrend_VolumeSpike"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ehlers Fisher Transform (9-period) on 6h close
    # Normalize price to [-1, 1] using 9-period min/max
    hl9 = pd.Series(high).rolling(window=9, min_periods=9).max() - pd.Series(low).rolling(window=9, min_periods=9).min()
    hl9 = hl9.replace(0, 1e-10)  # Avoid division by zero
    value1 = 0.66 * ((close - pd.Series(low).rolling(window=9, min_periods=9).min()) / hl9 - 0.5) + 0.67 * np.roll(
        0.66 * ((close - pd.Series(low).rolling(window=9, min_periods=9).min()) / hl9 - 0.5), 1)
    value1 = np.where(np.isnan(value1), 0, value1)
    value2 = np.roll(value1, 1)
    fish = 0.5 * np.log((1 + value2) / (1 - value2 + 1e-10)) + 0.5 * np.roll(
        0.5 * np.log((1 + value1) / (1 - value1 + 1e-10)), 1)
    fish = np.where(np.isnan(fish), 0, fish)
    
    # Volume filter: volume > 2.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and Fisher
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(fish[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Fisher crosses above -1.5 in uptrend with volume spike
            if (fish[i] > -1.5 and fish[i-1] <= -1.5 and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Fisher crosses below +1.5 in downtrend with volume spike
            elif (fish[i] < 1.5 and fish[i-1] >= 1.5 and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Fisher crosses below zero or trend reverses to downtrend
            if (fish[i] < 0 and fish[i-1] >= 0) or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Fisher crosses above zero or trend reverses to uptrend
            if (fish[i] > 0 and fish[i-1] <= 0) or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals