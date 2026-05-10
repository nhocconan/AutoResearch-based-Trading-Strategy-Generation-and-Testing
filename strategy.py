#/usr/bin/env python3
# 6H_4H_EMA_Crossover_12H_Volume_Regime_Filter
# Hypothesis: Uses 4h EMA cross for trend direction, filtered by 12h volume regime (low volatility) to avoid whipsaws.
# Enters long when 4h EMA(9) crosses above EMA(21) and 12h volume regime is calm (volume < 1.5x 50-period average).
# Enters short when 4h EMA(9) crosses below EMA(21) and 12h volume regime is calm.
# Exits on opposite EMA cross. Uses 6h timeframe for execution. Targets 12-30 trades per year with position size 0.25.
# Works in both bull/bear markets by using EMA cross with volatility filter to avoid false signals during high volatility.

name = "6H_4H_EMA_Crossover_12H_Volume_Regime_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA crossover
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(9) and EMA(21)
    ema_9_4h = pd.Series(df_4h['close']).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align EMA values to 6h timeframe
    ema_9_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_9_4h)
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Get 12h data for volume regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h volume 50-period average
    vol_ma_50_12h = pd.Series(df_12h['volume']).rolling(window=50, min_periods=50).mean().values
    vol_threshold_12h = vol_ma_50_12h * 1.5  # Volume regime: calm when volume < 1.5x average
    
    # Align volume threshold to 6h timeframe
    vol_threshold_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_threshold_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_9_4h_aligned[i]) or np.isnan(ema_21_4h_aligned[i]) or np.isnan(vol_threshold_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # EMA crossover signals
        ema_9_above_21 = ema_9_4h_aligned[i] > ema_21_4h_aligned[i]
        ema_9_below_21 = ema_9_4h_aligned[i] < ema_21_4h_aligned[i]
        
        # Volume regime filter: calm market (low volatility)
        vol_regime_calm = volume[i] < vol_threshold_12h_aligned[i]
        
        if position == 0:
            # Long entry: EMA(9) crosses above EMA(21) in calm volume regime
            if ema_9_above_21 and vol_regime_calm:
                signals[i] = 0.25
                position = 1
            # Short entry: EMA(9) crosses below EMA(21) in calm volume regime
            elif ema_9_below_21 and vol_regime_calm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: EMA(9) crosses below EMA(21)
            if ema_9_below_21:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: EMA(9) crosses above EMA(21)
            if ema_9_above_21:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals