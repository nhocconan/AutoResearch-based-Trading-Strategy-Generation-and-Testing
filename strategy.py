#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Uses Donchian channel from 4h for breakout signals, 1d EMA(50) for trend direction
# Volume spike (1.5x 20-period average) ensures participation and reduces false breakouts
# Only takes breakouts in the direction of the 1d trend to avoid counter-trend whipsaws
# Session filter (08-20 UTC) reduces noise trades outside active market hours
# Discrete position sizing 0.20 balances risk and minimizes fee churn
# Targets 15-37 trades/year (60-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by aligning with higher timeframe trend

name = "1h_Donchian20_1dTrend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_roll = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 4h Donchian channels to 1h
    high_roll_aligned = align_htf_to_ltf(prices, df_4h, high_roll)
    low_roll_aligned = align_htf_to_ltf(prices, df_4h, low_roll)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 1h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 1h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, EMA and volume MA)
    start_idx = 70  # max(20 for Donchian, 50 for EMA, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(high_roll_aligned[i]) or np.isnan(low_roll_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian AND uptrend AND volume confirm
            if (close[i] > high_roll_aligned[i] and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower Donchian AND downtrend AND volume confirm
            elif (close[i] < low_roll_aligned[i] and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR trend reverses to downtrend
            if (close[i] < low_roll_aligned[i] or 
                not uptrend):  # exited if price closes below 1d EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR trend reverses to uptrend
            if (close[i] > high_roll_aligned[i] or 
                not downtrend):  # exited if price closes above 1d EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals