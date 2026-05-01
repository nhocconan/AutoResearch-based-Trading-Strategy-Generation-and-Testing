#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation.
# Uses weekly EMA200 for trend filter to capture major market regime.
# Breakouts above/below 20-period Donchian channel are traded in direction of weekly trend.
# Volume confirmation ensures breakouts have institutional participation.
# Works in bull markets (buy breakouts with weekly uptrend) and bear markets (sell breakdowns with weekly downtrend).
# Discrete position sizing (0.25) limits drawdown. Target: 50-150 trades over 4 years.

name = "6h_Donchian20_Breakout_1wTrend_VolumeConfirm_v1"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 6h Donchian channel (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(200, 20) + 1  # 201 (for weekly EMA200 and Donchian)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: weekly EMA200 direction
        uptrend = curr_close > ema_200_1w_aligned[i]
        downtrend = curr_close < ema_200_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Donchian breakout conditions
        breakout_upper = curr_close > highest_high_20[i]  # Break above upper channel
        breakdown_lower = curr_close < lowest_low_20[i]   # Break below lower channel
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper channel AND weekly uptrend AND volume confirmation
            if breakout_upper and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower channel AND weekly downtrend AND volume confirmation
            elif breakdown_lower and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakdown below lower channel (reversal signal)
            if curr_close < lowest_low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above upper channel (reversal signal)
            if curr_close > highest_high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals