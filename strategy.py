#!/usr/bin/env python3
# 6h_VolatilityBreakout_With_VolumeAndTrendFilter
# Hypothesis: 6h volatility breakout (ATR-based) with volume confirmation and 1d EMA50 trend filter.
# Long when price breaks above recent high + volume spike + uptrend (price > 1d EMA50).
# Short when price breaks below recent low + volume spike + downtrend (price < 1d EMA50).
# Uses 10-bar lookback for breakout levels, 20-period ATR for volatility threshold.
# Volume spike: current volume > 1.5x 20-bar average volume.
# Designed to capture momentum bursts in both bull and bear markets while avoiding false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_VolatilityBreakout_With_VolumeAndTrendFilter"
timeframe = "6h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Parameters
    lookback = 10
    atr_period = 20
    vol_ma_period = 20
    vol_threshold = 1.5
    atr_multiplier = 0.5
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.full_like(high, np.nan)
    if len(high) >= atr_period:
        atr[atr_period] = np.nanmean(tr[1:atr_period+1])
        for i in range(atr_period + 1, len(high)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate rolling max/high and min/low for breakout levels
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback, len(high)):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Calculate volume moving average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= vol_ma_period:
        vol_ma[vol_ma_period] = np.mean(volume[:vol_ma_period])
        for i in range(vol_ma_period + 1, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * (vol_ma_period - 1) + volume[i]) / vol_ma_period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, atr_period, vol_ma_period, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * vol_threshold
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i] + atr[i] * atr_multiplier
        breakout_down = close[i] < lowest_low[i] - atr[i] * atr_multiplier
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume spike + upward breakout
            if uptrend and vol_spike and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume spike + downward breakout
            elif downtrend and vol_spike and breakout_down:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit on trend reversal or volatility contraction
            if not uptrend or not vol_spike or close[i] < highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit on trend reversal or volatility contraction
            if not downtrend or not vol_spike or close[i] > lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals