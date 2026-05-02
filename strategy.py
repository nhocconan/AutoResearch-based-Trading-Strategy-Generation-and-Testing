#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# Uses Donchian channel from 4h for breakout signals, 12h EMA(50) for trend direction
# Volume spike (1.5x 20-period average) ensures participation and reduces false breakouts
# Only takes breakouts in the direction of the 12h trend to avoid counter-trend whipsaws
# Discrete position sizing 0.30 balances risk and minimizes fee churn
# Targets 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by aligning with higher timeframe trend

name = "4h_Donchian20_12hTrend_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate 4h Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 4h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, EMA and volume MA)
    start_idx = 70  # max(20 for Donchian, 50 for EMA, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian AND uptrend AND volume confirm
            if (close[i] > high_roll[i] and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below lower Donchian AND downtrend AND volume confirm
            elif (close[i] < low_roll[i] and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR trend reverses to downtrend
            if (close[i] < low_roll[i] or 
                not uptrend):  # exited if price closes below 12h EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR trend reverses to uptrend
            if (close[i] > high_roll[i] or 
                not downtrend):  # exited if price closes above 12h EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals