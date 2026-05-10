#!/usr/bin/env python3
# 1D_SwingFailure_Rebound_WithVolume
# Hypothesis: On daily timeframe, enter long when price makes a new low but closes above prior day's low (bullish failure test) with volume expansion.
# Enter short when price makes a new high but closes below prior day's high (bearish failure test) with volume expansion.
# Uses 1-week trend filter to align with higher timeframe momentum and avoid counter-trend trades.
# Target: 10-25 trades/year per symbol (40-100 total over 4 years).

name = "1D_SwingFailure_Rebound_WithVolume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for swing failure detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Bullish failure: new low but close above prior low
    new_low = low_1d < np.roll(low_1d, 1)
    close_above_prior_low = close_1d > np.roll(low_1d, 1)
    bullish_failure = new_low & close_above_prior_low
    
    # Bearish failure: new high but close below prior high
    new_high = high_1d > np.roll(high_1d, 1)
    close_below_prior_high = close_1d < np.roll(high_1d, 1)
    bearish_failure = new_high & close_below_prior_high
    
    # Volume confirmation: current volume > 1.8x 20-day average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.8)
    
    # 1-week trend filter: EMA(50) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_1w > ema_50
    
    # Align 1d indicators to daily (no alignment needed as we're on 1d timeframe)
    # But we need to align weekly trend to daily
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish failure + volume confirmation + weekly uptrend
            if i < len(bullish_failure) and bullish_failure[i] and volume_confirm[i] and trend_up_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish failure + volume confirmation + weekly downtrend
            elif i < len(bearish_failure) and bearish_failure[i] and volume_confirm[i] and not trend_up_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish failure or trend change
            if i < len(bearish_failure) and bearish_failure[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish failure or trend change
            if i < len(bullish_failure) and bullish_failure[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals