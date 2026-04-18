#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_Filter
1d strategy using Kaufman Adaptive Moving Average (KAMA) for trend detection,
combined with volume confirmation and RSI filter to reduce whipsaws.
- Long: KAMA trending up + volume > 1.5x 20-day avg + RSI > 50
- Short: KAMA trending down + volume > 1.5x 20-day avg + RSI < 50
- Exit: Opposite KAMA direction signal
Designed for ~10-20 trades/year per symbol (40-80 total over 4 years)
KAMA adapts to market noise, reducing false signals in ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA and volume average
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Parameters: ER period=10, Fast EMA=2, Slow EMA=30
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # This is wrong, need to fix
    
    # Correct ER calculation
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    # Volatility is sum of absolute changes over er_period
    volatility = np.zeros_like(close_1d)
    for i in range(er_period, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-er_period:i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(close_1d)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_ema + 1)
    slow_sc = 2 / (slow_ema + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI calculation (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Subsequent averages
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close_1d)
    rsi = np.zeros_like(close_1d)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[avg_loss == 0] = 100  # No loss = RSI 100
    
    # Daily volume average (20-period)
    vol_ma_20 = np.zeros_like(volume_1d)
    for i in range(len(volume_1d)):
        if i < 20:
            vol_ma_20[i] = np.nan
        else:
            vol_ma_20[i] = np.mean(volume_1d[i-20:i])
    
    # Align all daily data to 1d timeframe (identity since same timeframe)
    kama_aligned = kama  # Same timeframe, no alignment needed
    rsi_aligned = rsi    # Same timeframe, no alignment needed
    vol_ma_aligned = vol_ma_20  # Same timeframe, no alignment needed
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for KAMA and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend direction (comparing current to previous)
        kama_up = kama_aligned[i] > kama_aligned[i-1]
        kama_down = kama_aligned[i] < kama_aligned[i-1]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # RSI filter
        rsi_filter_long = rsi_aligned[i] > 50
        rsi_filter_short = rsi_aligned[i] < 50
        
        if position == 0:
            # Long: KAMA up + volume + RSI > 50
            if kama_up and vol_confirm and rsi_filter_long:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + volume + RSI < 50
            elif kama_down and vol_confirm and rsi_filter_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down
            if kama_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up
            if kama_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0