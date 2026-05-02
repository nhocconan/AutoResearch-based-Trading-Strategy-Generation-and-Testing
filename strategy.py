#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation (2x 20-period average)
# Uses Donchian channels for structure, 1d EMA for trend alignment, volume spike for participation
# Only takes breakouts in direction of 1d trend to avoid counter-trend whipsaws
# Discrete position sizing 0.25 to balance risk and minimize fee churn
# Targets 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by aligning with higher timeframe trend

name = "4h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 4h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 4h volume confirmation (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and volume MA)
    start_idx = 40  # 20 for Donchian + 20 for volume + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian AND uptrend AND volume confirm
            if (close[i] > high_ma[i] and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND downtrend AND volume confirm
            elif (close[i] < low_ma[i] and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR trend reverses to downtrend
            if (close[i] < low_ma[i] or 
                not uptrend):  # exited if price closes below 1d EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR trend reverses to uptrend
            if (close[i] > high_ma[i] or 
                not downtrend):  # exited if price closes above 1d EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals