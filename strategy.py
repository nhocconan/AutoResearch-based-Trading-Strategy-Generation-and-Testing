#!/usr/bin/env python3
"""
6h_EMA_Crossover_VolumeRegime_HTFTrend
Hypothesis: 6h EMA(9,21) crossover with volume regime filter (volume > 1.5x 50-period median) and 1d trend filter (price > EMA50).
Enters long on bullish crossover when volume is elevated and 1d trend is bullish.
Enters short on bearish crossover when volume is elevated and 1d trend is bearish.
Uses discrete position sizing (0.25) to minimize churn. Designed for 50-150 total trades over 4 years.
Works in both bull and bear markets by following 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMAs on primary timeframe (6h)
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume regime: volume > 1.5x 50-period median (using rolling median via percentile)
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_regime = volume > (1.5 * vol_median)
    
    # Load 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 21-period EMA and 50-period volume median)
    start_idx = max(21, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Bullish crossover: EMA9 crosses above EMA21
        bullish_cross = ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1]
        # Bearish crossover: EMA9 crosses below EMA21
        bearish_cross = ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1]
        
        # Long logic: bullish crossover + volume regime + bullish 1d trend
        if bullish_cross and volume_regime[i] and close[i] > ema_50_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: bearish crossover + volume regime + bearish 1d trend
        elif bearish_cross and volume_regime[i] and close[i] < ema_50_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite crossover
        elif position == 1 and bearish_cross:
            signals[i] = 0.0
            position = 0
        elif position == -1 and bullish_cross:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_EMA_Crossover_VolumeRegime_HTFTrend"
timeframe = "6h"
leverage = 1.0