#!/usr/bin/env python3
"""
1d_adx_rsi_momentum_v1
Hypothesis: Combine ADX trend strength with RSI momentum to capture strong moves in both bull and bear markets.
Long when ADX > 25 (strong trend) and RSI > 50 (bullish momentum).
Short when ADX > 25 (strong trend) and RSI < 50 (bearish momentum).
Use weekly timeframe for trend filter to avoid whipsaws.
Designed for low trade frequency (target: 10-30 trades/year) to minimize fee drag on daily timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_adx_rsi_momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/14)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    # Initial values
    atr[13] = np.mean(tr[1:14])
    plus_dm_sum = np.sum(plus_dm[1:14])
    minus_dm_sum = np.sum(minus_dm[1:14])
    
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / 14) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / 14) + minus_dm[i]
        
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(14, n):
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum if di_sum != 0 else 0
    
    # Smooth DX to get ADX
    adx[27] = np.mean(dx[14:28])  # First ADX value after 2*14 periods
    for i in range(28, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Weekly trend filter: EMA(50) on weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = np.zeros(len(close_1w))
    ema_50[49] = np.mean(close_1w[:50])
    for i in range(50, len(close_1w)):
        ema_50[i] = (close_1w[i] * 2 + ema_50[i-1] * 49) / 50
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Generate signals
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Long: strong uptrend (ADX > 25) + bullish momentum (RSI > 50) + price above weekly EMA
        if adx[i] > 25 and rsi[i] > 50 and close[i] > ema_50_aligned[i]:
            signals[i] = 0.25
        # Short: strong downtrend (ADX > 25) + bearish momentum (RSI < 50) + price below weekly EMA
        elif adx[i] > 25 and rsi[i] < 50 and close[i] < ema_50_aligned[i]:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals