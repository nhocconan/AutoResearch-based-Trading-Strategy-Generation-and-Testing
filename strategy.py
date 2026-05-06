#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot point breakout with 1d trend filter and volume confirmation
# Long when price breaks above weekly R1 AND 1d close > 1d open (bullish daily candle) AND 6h volume > 1.5 * avg_volume(20)
# Short when price breaks below weekly S1 AND 1d close < 1d open (bearish daily candle) AND 6h volume > 1.5 * avg_volume(20)
# Exit when price returns to weekly pivot point (PP)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 80-120 total trades over 4 years (20-30/year) for 6h timeframe
# Weekly pivot provides key institutional levels from prior week
# 1d candle direction ensures alignment with recent daily momentum
# Volume confirmation filters out low-conviction breakouts
# Works in bull markets (breakout continuations) and bear markets (breakdown continuations)

name = "6h_WeeklyPivot_R1S1_Breakout_1dCandleDir_Volume"
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
    open_price = prices['open'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get weekly data ONCE before loop for pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:  # Need at least 1 completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point (PP), R1, S1
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = (2 * pp_1w) - low_1w
    s1_1w = (2 * pp_1w) - high_1w
    
    # Align weekly pivot levels to 6h timeframe (wait for completed weekly bar)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Get 1d data ONCE before loop for candle direction filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:  # Need at least 1 completed daily bar
        return np.zeros(n)
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d candle direction: 1 for bullish (close > open), -1 for bearish (close < open), 0 for doji
    candle_dir_1d = np.where(close_1d > open_1d, 1, np.where(close_1d < open_1d, -1, 0))
    candle_dir_1d_aligned = align_htf_to_ltf(prices, df_1d, candle_dir_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(candle_dir_1d_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1, bullish daily candle, volume spike
            if (close[i] > r1_1w_aligned[i] and close[i-1] <= r1_1w_aligned[i-1] and 
                candle_dir_1d_aligned[i] == 1 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1, bearish daily candle, volume spike
            elif (close[i] < s1_1w_aligned[i] and close[i-1] >= s1_1w_aligned[i-1] and 
                  candle_dir_1d_aligned[i] == -1 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly pivot point (PP)
            if close[i] <= pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly pivot point (PP)
            if close[i] >= pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals