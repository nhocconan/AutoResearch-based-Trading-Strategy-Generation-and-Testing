#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume confirmation
# Uses Donchian channel from 6h for breakout signals, 1w EMA(34) for trend direction
# Volume spike (1.8x 20-period average) ensures participation and reduces false breakouts
# Only takes breakouts in the direction of the 1w trend to avoid counter-trend whipsaws
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by aligning with higher timeframe trend
# 1w trend filter provides strong directional bias suitable for 6h timeframe

name = "6h_Donchian20_1wTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate 6h Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 6h volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, EMA and volume MA)
    start_idx = 70  # max(20 for Donchian, 34 for EMA, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian AND uptrend AND volume confirm
            if (close[i] > high_roll[i] and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND downtrend AND volume confirm
            elif (close[i] < low_roll[i] and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR trend reverses to downtrend
            if (close[i] < low_roll[i] or 
                not uptrend):  # exited if price closes below 1w EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR trend reverses to uptrend
            if (close[i] > high_roll[i] or 
                not downtrend):  # exited if price closes above 1w EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals