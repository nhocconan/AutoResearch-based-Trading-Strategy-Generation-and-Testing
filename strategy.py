#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
# Uses 1w EMA200 for higher timeframe trend filter (bull/bear regime).
# Volume confirmation (>1.5x 50-bar avg) reduces false breakouts.
# Session filter (00-23 UTC) always active for 12h timeframe.
# Discrete position sizing at ±0.25 to balance return and fee drag.
# Target: 80-120 total trades over 4 years (20-30/year) to avoid fee drag on 12h timeframe.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion exits when price retests Donchian levels.

name = "12h_Donchian20_1wEMA200_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 210:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Donchian(20) channels from 1d data for structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) upper/lower bands
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Volume confirmation: volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.5 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_200_1w = ema_200_1w_aligned[i]
        curr_upper = upper_20_aligned[i]
        curr_lower = lower_20_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian, close > 1w EMA200, volume spike
            if (curr_close > curr_upper and 
                curr_close > curr_ema_200_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, close < 1w EMA200, volume spike
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_200_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price moves back below upper Donchian (mean reversion)
            if curr_close < curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price moves back above lower Donchian (mean reversion)
            if curr_close > curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals