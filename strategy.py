#!/usr/bin/env python3
"""
6h_Stochastic_RSI_Trend
Hypothesis: Combines Stochastic RSI momentum with 1d trend filter and volume confirmation to capture medium-term swings.
Designed for 6h timeframe with selective entries (target: 20-40 trades/year) to minimize fee drag.
Works in both bull and bear markets by using trend filter to align with higher timeframe direction.
"""

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
    
    # Stochastic RSI parameters
    rsi_period = 14
    stoch_period = 14
    k_smooth = 3
    d_smooth = 3
    
    # Calculate RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Wilder's smoothing
    for i in range(rsi_period, n):
        if i == rsi_period:
            avg_gain[i] = np.mean(gain[i-rsi_period+1:i+1])
            avg_loss[i] = np.mean(loss[i-rsi_period+1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_min = np.full(n, np.nan)
    rsi_max = np.full(n, np.nan)
    for i in range(stoch_period, n):
        rsi_min[i] = np.min(rsi[i-stoch_period+1:i+1])
        rsi_max[i] = np.max(rsi[i-stoch_period+1:i+1])
    
    stoch_rsi = np.divide((rsi - rsi_min), (rsi_max - rsi_min), 
                          out=np.full_like(rsi, np.nan), where=(rsi_max - rsi_min)!=0)
    stoch_rsi = stoch_rsi * 100  # Convert to 0-100 scale
    
    # %K and %D lines
    k = np.full(n, np.nan)
    d = np.full(n, np.nan)
    
    for i in range(k_smooth, n):
        k[i] = np.mean(stoch_rsi[i-k_smooth+1:i+1])
    
    for i in range(d_smooth, n):
        d[i] = np.mean(k[i-d_smooth+1:i+1])
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    k_ema = 2 / (ema_period + 1)
    
    for i in range(ema_period, len(close_1d)):
        if i == ema_period:
            ema_1d[i] = np.mean(close_1d[i-ema_period+1:i+1])
        else:
            ema_1d[i] = close_1d[i] * k_ema + ema_1d[i-1] * (1 - k_ema)
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, rsi_period + stoch_period + k_smooth + d_smooth, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(k[i]) or np.isnan(d[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Stochastic RSI oversold (<20) and rising, with uptrend and volume spike
            if k[i] < 20 and d[i] < 20 and k[i] > d[i] and ema_1d_aligned[i] < close[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Stochastic RSI overbought (>80) and falling, with downtrend and volume spike
            elif k[i] > 80 and d[i] > 80 and k[i] < d[i] and ema_1d_aligned[i] > close[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Stochastic RSI overbought or trend weakens
            if k[i] > 80 or d[i] > 80 or ema_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Stochastic RSI oversold or trend weakens
            if k[i] < 20 or d[i] < 20 or ema_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Stochastic_RSI_Trend"
timeframe = "6h"
leverage = 1.0