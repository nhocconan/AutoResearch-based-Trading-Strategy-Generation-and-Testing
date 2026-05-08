#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Trend_With_RSI_And_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for KAMA, RSI and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0) if False else None
    # Correct volatility calculation: sum of absolute changes over er_length period
    volatility = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < er_length:
            volatility[i] = np.nan
        else:
            volatility[i] = np.sum(np.abs(np.diff(close_1d[i-er_length:i+1])))
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI calculation
    rsi_length = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i < rsi_length:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        elif i == rsi_length:
            avg_gain[i] = np.mean(gain[1:rsi_length+1])
            avg_loss[i] = np.mean(loss[1:rsi_length+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_length-1) + gain[i]) / rsi_length
            avg_loss[i] = (avg_loss[i-1] * (rsi_length-1) + loss[i]) / rsi_length
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection: current volume > 2.0 * 20-day average
    vol_ma20 = np.full_like(volume_1d, np.nan)
    for i in range(len(volume_1d)):
        if i < 20:
            vol_ma20[i] = np.nan
        else:
            vol_ma20[i] = np.mean(volume_1d[i-20:i])
    
    volume_spike = volume_1d > (vol_ma20 * 2.0)
    
    # Align all indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above KAMA, RSI > 50, volume spike
            long_cond = (close[i] > kama_aligned[i] and 
                        rsi_aligned[i] > 50 and 
                        volume_spike_aligned[i])
            
            # Short entry: price below KAMA, RSI < 50, volume spike
            short_cond = (close[i] < kama_aligned[i] and 
                         rsi_aligned[i] < 50 and 
                         volume_spike_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA adapts to market conditions - faster in trends, slower in ranges.
# Combined with RSI for momentum confirmation and volume spike for validation.
# Works in both bull and bear markets by adapting to volatility.
# Target: 15-25 trades/year to minimize fee decay while capturing significant moves.