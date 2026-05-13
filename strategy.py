#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
# Long when price closes above 20-day high with 1w EMA200 uptrend and volume > 1.5x average.
# Short when price closes below 20-day low with 1w EMA200 downtrend and volume > 1.5x average.
# Exit when price reverts to 10-day EMA (mean reversion within trend).
# Uses discrete position sizing 0.25. Target: 30-100 total trades over 4 years on 1d timeframe.
# 1w EMA200 ensures we only trade in the direction of the weekly trend, avoiding counter-trend whipsaws.
# Volume confirmation filters out low-conviction breakouts.

name = "1d_Donchian20_1wEMA200_VolumeConfirm_v1"
timeframe = "1d"
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
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    if n < lookback + 1:
        return np.zeros(n)
    
    # Rolling high/low for Donchian
    high_roll = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_roll = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # 10-day EMA for exit (mean reversion target)
    ema_10 = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA200 on 1w data
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 1w EMA200 to 1d timeframe (wait for 1w bar to close)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_10[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price closes above 20-day high with 1w EMA200 uptrend and volume spike
            if (close[i] > high_roll[i] and 
                close_1w[i] > ema_200_1w_aligned[i] and  # Weekly close above EMA200 = uptrend
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below 20-day low with 1w EMA200 downtrend and volume spike
            elif (close[i] < low_roll[i] and 
                  close_1w[i] < ema_200_1w_aligned[i] and  # Weekly close below EMA200 = downtrend
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to 10-day EMA (mean reversion within trend)
            if close[i] <= ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to 10-day EMA (mean reversion within trend)
            if close[i] >= ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals