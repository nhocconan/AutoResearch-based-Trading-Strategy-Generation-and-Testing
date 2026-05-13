#!/usr/bin/env python3
# Hypothesis: 12h Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In ranging markets (2025+ test period),
# mean reversion at extreme levels works well. 1w EMA50 ensures we only take mean reverts
# in the direction of the higher timeframe trend to avoid fighting the trend.
# Volume confirmation ensures participation. Discrete sizing (0.25) minimizes fee churn.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.

name = "12h_WilliamsR_MeanReversion_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_vol, 14), n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80) and price above 1w EMA50 (uptrend)
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25  # Long position
                position = 1
            # SHORT: Williams %R overbought (> -20) and price below 1w EMA50 (downtrend)
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25  # Short position
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to neutral (> -50) or trend changes
            if williams_r_aligned[i] > -50 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0  # Exit long
                position = 0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:
            # EXIT SHORT: Williams %R returns to neutral (< -50) or trend changes
            if williams_r_aligned[i] < -50 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0  # Exit short
                position = 0
            else:
                signals[i] = -0.25  # Maintain short
    
    return signals