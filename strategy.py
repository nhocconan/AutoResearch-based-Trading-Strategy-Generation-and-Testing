#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to determine trend direction, combined with RSI for momentum and Choppiness Index to filter ranging markets. Only take trades when KAMA slope confirms trend, RSI is in overbought/oversold territory but with divergence, and market is trending (CHOP < 38.2). This avoids whipsaws in ranging markets while capturing strong trends. Works in both bull and bear markets by following adaptive trend. Targets 15-25 trades/year on daily timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for regime filter
    df_1w = get_htf_data(prices, '1w')
    
    # KAMA calculation (ER = 10, Fast = 2, Slow = 30)
    kama_period = 10
    fast_sc = 2
    slow_sc = 30
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, kama_period))  # |close[t] - close[t-kama_period]|
    abs_change = np.abs(np.diff(close))  # |close[t] - close[t-1]|
    
    # Sum of absolute changes over kama_period
    sum_abs_change = np.zeros_like(close)
    for i in range(kama_period, len(close)):
        sum_abs_change[i] = np.sum(abs_change[i-kama_period+1:i+1])
    
    # Avoid division by zero
    er = np.zeros_like(close)
    mask = sum_abs_change > 0
    er[kama_period:] = change[kama_period:] / sum_abs_change[kama_period:]
    
    # Smoothing constants
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[kama_period] = close[kama_period]  # Initialize with simple average
    for i in range(kama_period + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (trend direction)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # RSI calculation (14 period)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    if len(close) >= rsi_period:
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    # Calculate RSI
    rs = np.zeros_like(close)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index on weekly data
    chop_period = 14
    if len(df_1w) >= chop_period:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = np.abs(high_1w - low_1w)
        tr2 = np.abs(high_1w - np.roll(close_1w, 1))
        tr3 = np.abs(low_1w - np.roll(close_1w, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_1w[0] - low_1w[0]
        
        # Sum of TR over chop_period
        sum_tr = np.zeros_like(tr)
        for i in range(chop_period, len(tr)):
            sum_tr[i] = np.sum(tr[i-chop_period+1:i+1])
        
        # Highest high and lowest low over chop_period
        max_high = np.zeros_like(high_1w)
        min_low = np.zeros_like(low_1w)
        for i in range(chop_period, len(high_1w)):
            max_high[i] = np.max(high_1w[i-chop_period+1:i+1])
            min_low[i] = np.min(low_1w[i-chop_period+1:i+1])
        
        # Avoid division by zero
        range_max_min = max_high - min_low
        chop = np.zeros_like(tr)
        mask = (range_max_min > 0) & (sum_tr > 0)
        chop[mask] = 100 * np.log10(sum_tr[mask] / range_max_min[mask]) / np.log10(chop_period)
        
        # Align chop to daily
        chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    else:
        chop_aligned = np.full(n, 50.0)  # Neutral value if insufficient data
    
    # Align KAMA and RSI to daily (they're already calculated on daily)
    kama_aligned = kama
    kama_slope_aligned = kama_slope
    rsi_aligned = rsi
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kama_period + 1, rsi_period + 1, chop_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(kama_slope[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when market is trending (CHOP < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: KAMA rising, RSI > 50 (bullish momentum), and trending market
            if kama_slope[i] > 0 and rsi[i] > 50 and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI < 50 (bearish momentum), and trending market
            elif kama_slope[i] < 0 and rsi[i] < 50 and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA slope turns negative or RSI < 40 (loss of momentum)
            if kama_slope[i] <= 0 or rsi[i] < 40:
                signals[i] = 0.0  # flat
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA slope turns positive or RSI > 60 (loss of momentum)
            if kama_slope[i] >= 0 or rsi[i] > 60:
                signals[i] = 0.0  # flat
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0