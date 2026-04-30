#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h/1d trend filter and session filter
# Uses fast EMA(9) and slow EMA(21) on 1h for entry timing
# 4h EMA50 and 1d EMA200 ensure alignment with higher timeframe trends
# Session filter (08-20 UTC) reduces noise trades during low-volume periods
# Volume confirmation (>1.5x average) ensures participation
# Discrete sizing (0.20) minimizes fee churn; target 60-150 total trades over 4 years
# Works in bull/bear: EMA crossover captures momentum, HTF filters avoid counter-trend trades

name = "1h_EMA9_21_Crossover_4hEMA50_1dEMA200_Session_Filter_v1"
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate EMAs for 1h
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(9, 21, 50, 200, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_ema_fast = ema_fast[i]
        curr_ema_slow = ema_slow[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_ema_200_1d = ema_200_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        prev_ema_fast = ema_fast[i-1]
        prev_ema_slow = ema_slow[i-1]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and HTF trend alignment
            if curr_volume_confirm:
                # Bullish entry: fast EMA crosses above slow EMA + above both HTF EMAs
                if (prev_ema_fast <= prev_ema_slow and curr_ema_fast > curr_ema_slow and
                    curr_ema_fast > curr_ema_50_4h and curr_ema_fast > curr_ema_200_1d):
                    signals[i] = 0.20
                    position = 1
                # Bearish entry: fast EMA crosses below slow EMA + below both HTF EMAs
                elif (prev_ema_fast >= prev_ema_slow and curr_ema_fast < curr_ema_slow and
                      curr_ema_fast < curr_ema_50_4h and curr_ema_fast < curr_ema_200_1d):
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: fast EMA crosses below slow EMA
            if prev_ema_fast >= prev_ema_slow and curr_ema_fast < curr_ema_slow:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: fast EMA crosses above slow EMA
            if prev_ema_fast <= prev_ema_slow and curr_ema_fast > curr_ema_slow:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals