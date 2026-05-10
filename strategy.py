#!/usr/bin/env python3
# 4h_RSI2_TrendFilter_Volume
# Hypothesis: RSI(2) identifies extreme short-term overbought/oversold conditions.
# In trending markets (identified by EMA50 slope on 1d), these extremes offer high-probability
# mean-reversion entries. Volume filter ensures participation. Works in bull via buying dips in uptrend
# and in bear via selling rallies in downtrend. Low trade frequency due to strict RSI thresholds.
# Target: 20-40 trades/year on 4h timeframe.

name = "4h_RSI2_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period):
    """Calculate RSI with given period."""
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on daily timeframe for trend
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    ema50_1d_slope = ema50_1d.diff()  # Positive = uptrend, Negative = downtrend
    
    # Align EMA50 and its slope to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d.values)
    ema50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_slope.values)
    
    # Get 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(2) on 4h
    rsi2 = rsi(close, 2)
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50) + RSI(2) (2) + vol EMA (20)
    start_idx = max(50, 2, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_1d_slope_aligned[i]) or
            np.isnan(rsi2[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from EMA50 slope
        uptrend = ema50_1d_slope_aligned[i] > 0
        downtrend = ema50_1d_slope_aligned[i] < 0
        
        if position == 0:
            # Long: RSI2 < 10 (oversold) AND uptrend AND volume
            if rsi2[i] < 10 and uptrend and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI2 > 90 (overbought) AND downtrend AND volume
            elif rsi2[i] > 90 and downtrend and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI2 > 50 (neutral) OR trend turns down
            if rsi2[i] > 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI2 < 50 (neutral) OR trend turns up
            if rsi2[i] < 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals