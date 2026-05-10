#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI_MeanReversion
# Hypothesis: KAMA adapts to market efficiency, providing reliable trend direction.
# In trending markets (KAMA slope > 0), buy pullbacks when RSI < 40.
# In ranging markets (KAMA slope ≈ 0), sell bounces when RSI > 60.
# Uses 4h timeframe with daily trend filter for institutional alignment.
# Works in bull markets via trend-following pullbacks, bear markets via mean-reversion selling.
# Low trade frequency expected due to dual-condition requirement.

name = "4h_KAMA_Direction_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    er = np.zeros_like(close)
    for i in range(er_period, len(close)):
        if volatility[i] != 0:
            er[i] = change[i-er_period:i+1].sum() / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama_vals = np.zeros_like(close)
    kama_vals[0] = close[0]
    for i in range(1, len(close)):
        kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
    return kama_vals

def rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    for i in range(1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_vals = 100 - (100 / (1 + rs))
    return rsi_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily close for trend alignment
    close_1d = df_1d['close'].values
    
    # Get 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on 4h
    kama_vals = kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # Calculate RSI on 4h
    rsi_vals = rsi(close, period=14)
    
    # Calculate KAMA slope (trend direction)
    kama_slope = np.zeros_like(kama_vals)
    for i in range(1, len(kama_vals)):
        kama_slope[i] = kama_vals[i] - kama_vals[i-1]
    
    # Align daily close to 4h for trend filter
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10) + RSI (14) + volume EMA (20)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_vals[i]) or 
            np.isnan(rsi_vals[i]) or
            np.isnan(kama_slope[i]) or
            np.isnan(close_1d_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market state: trending vs ranging
        # Trending: significant KAMA slope
        trending = np.abs(kama_slope[i]) > (close[i] * 0.001)  # 0.1% of price
        ranging = np.abs(kama_slope[i]) <= (close[i] * 0.001)
        
        if position == 0:
            # In trending market: buy pullbacks (KAMA up, RSI low)
            if trending and kama_slope[i] > 0 and rsi_vals[i] < 40 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # In ranging market: sell bounces (RSI high)
            elif ranging and rsi_vals[i] > 60 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or trend turns down
            if rsi_vals[i] > 70 or kama_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold or trend turns up
            if rsi_vals[i] < 30 or kama_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals