#!/usr/bin/env python3
"""
1d_1w_adx_rsi_momentum
Uses weekly ADX for trend strength and daily RSI for momentum.
Enters long when weekly ADX > 25 (trending) and daily RSI crosses above 50 from below.
Enters short when weekly ADX > 25 and daily RSI crosses below 50 from above.
Exits when RSI crosses back to 50 or ADX falls below 20.
Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
Works in trending markets by following momentum with trend filter.
"""

name = "1d_1w_adx_rsi_momentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    return np.maximum(tr1, np.maximum(tr2, tr3))

def adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    
    # Calculate True Range
    tr = true_range(high, low, close)
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth the values
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(tr)
    minus_di = np.zeros_like(tr)
    
    # Initial values
    atr[period-1] = np.mean(tr[:period])
    plus_dm_sum = np.sum(plus_dm[:period])
    minus_dm_sum = np.sum(minus_dm[:period])
    
    # Smooth subsequent values
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
    
    # Calculate DX and ADX
    dx = np.zeros_like(tr)
    adx_val = np.zeros_like(tr)
    
    for i in range(period, len(tr)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # Smooth DX to get ADX
    adx_val[2*period-2] = np.mean(dx[period-1:2*period-1])
    for i in range(2*period-1, len(tr)):
        adx_val[i] = (adx_val[i-1] * (period-1) + dx[i]) / period
    
    return adx_val

def rsi(close, period=14):
    """Calculate RSI"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.zeros_like(close)
    rsi_val = np.zeros_like(close)
    
    for i in range(period, len(close)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi_val[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_val[i] = 100
    
    return rsi_val

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ADX
    adx_1w = adx(high_1w, low_1w, close_1w, period=14)
    
    # Align ADX to daily timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate daily RSI
    rsi_1d = rsi(close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if np.isnan(adx_1w_aligned[i]) or np.isnan(rsi_1d[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: weekly ADX > 25 (trending) and daily RSI crosses above 50 from below
        if (adx_1w_aligned[i] > 25 and 
            rsi_1d[i] > 50 and 
            i > 0 and rsi_1d[i-1] <= 50 and
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: weekly ADX > 25 and daily RSI crosses below 50 from above
        elif (adx_1w_aligned[i] > 25 and 
              rsi_1d[i] < 50 and 
              i > 0 and rsi_1d[i-1] >= 50 and
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: RSI crosses back to 50 or ADX falls below 20
        elif position == 1 and (rsi_1d[i] < 50 or adx_1w_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_1d[i] > 50 or adx_1w_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals