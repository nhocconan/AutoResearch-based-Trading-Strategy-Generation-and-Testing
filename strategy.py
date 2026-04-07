#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour KAMA with daily trend filter and volume confirmation
# KAMA adapts to market noise - fast in trending, slow in ranging markets
# Daily trend filter (EMA50) ensures we only trade in direction of higher timeframe trend
# Volume confirms institutional participation at entry
# Designed for low frequency: 15-30 trades per year to minimize fee drag in 6h timeframe

name = "6h_kama_daily_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate KAMA on 6h data
    # Efficiency Ratio: |change| / sum(|changes|) over 10 periods
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # Avoid division by zero
    sum_abs_change = np.zeros(n)
    for i in range(10, n):
        sum_abs_change[i] = np.sum(abs_change[i-9:i+1])
    
    er = np.zeros(n)
    for i in range(10, n):
        if sum_abs_change[i] > 0:
            er[i] = change[i] / sum_abs_change[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if price crosses below KAMA (trend change) or breaks daily EMA50
            if close[i] < kama[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if price crosses above KAMA (trend change) or breaks daily EMA50
            if close[i] > kama[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price above KAMA AND uptrend AND volume confirmation
            if close[i] > kama[i] and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price below KAMA AND downtrend AND volume confirmation
            elif close[i] < kama[i] and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals