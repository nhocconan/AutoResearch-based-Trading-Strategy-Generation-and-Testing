#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Trend_Momentum_v2
Hypothesis: Uses KAMA trend direction on daily timeframe with RSI momentum and volume confirmation for entries.
Trades only in direction of KAMA trend to avoid whipsaws. Targets 8-15 trades/year per symbol with high-probability setups.
Works in bull markets via trend continuation and in bear markets via mean-reversion bounces off trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_Trend_Momentum_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for KAMA calculation (same as primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recalculate volatility properly
    volatility = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    # For first element, set to small value to avoid division by zero
    if len(volatility) > 0:
        volatility[0] = np.abs(close_1d[0] - close_1d[0]) if len(close_1d) > 0 else 1e-10
    
    # Avoid division by zero
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    if len(close_1d) > 0:
        kama[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (no alignment needed as same timeframe)
    kama_aligned = kama  # Already on 1d timeframe
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    if len(gain) > 0:
        avg_gain[0] = gain[0]
        avg_loss[0] = loss[0]
        for i in range(1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # Handle division by zero
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    # Handle edges
    for i in range(min(10, len(volume))):
        vol_ma_20[i] = np.mean(volume[:i+10]) if i+10 < len(volume) else np.mean(volume[max(0, i-10):i+1])
    for i in range(max(0, len(volume)-10), len(volume)):
        vol_ma_20[i] = np.mean(volume[max(0, i-10):]) if i-10 >= 0 else np.mean(volume[:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Trend filter: price relative to KAMA
        above_kama = close[i] > kama_aligned[i]
        below_kama = close[i] < kama_aligned[i]
        
        # RSI conditions: momentum confirmation
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        rsi_momentum_up = rsi[i] > 50 and rsi[i] > rsi[i-1] if i > 0 else False
        rsi_momentum_down = rsi[i] < 50 and rsi[i] < rsi[i-1] if i > 0 else False
        
        # Entry conditions
        long_entry = above_kama and volume_filter and (rsi_momentum_up or rsi_oversold)
        short_entry = below_kama and volume_filter and (rsi_momentum_down or rsi_overbought)
        
        # Exit conditions: opposite condition or loss of momentum
        long_exit = below_kama or (rsi[i] < 40 and rsi[i] < rsi[i-1]) if i > 0 else below_kama
        short_exit = above_kama or (rsi[i] > 60 and rsi[i] > rsi[i-1]) if i > 0 else above_kama
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals