#!/usr/bin/env python3
"""
4h_RSI40_60_MeanReversion_Range
Hypothesis: In ranging markets (identified by low ADX), RSI extremes (below 40 or above 60) 
provide mean-reversion opportunities. Uses 4h timeframe for lower trade frequency and 
ADX(14) < 20 as range filter. Targets 20-40 trades/year with disciplined entries.
Works in both bull and bear markets by avoiding strong trends and focusing on mean reversion 
within ranges.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    
    # Wilder's smoothing
    for i in range(1, n):
        if i == 1:
            avg_gain[i] = gain[0] if len(gain) > 0 else 0
            avg_loss[i] = loss[0] if len(loss) > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # ADX(14) for range detection
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed averages
    atr = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    adx = np.full(n, np.nan)
    
    for i in range(1, n):
        if i < 14:
            if i == 1:
                atr[i] = tr[0]
                plus_di[i] = plus_dm[0] if len(plus_dm) > 0 else 0
                minus_di[i] = minus_dm[0] if len(minus_dm) > 0 else 0
            else:
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
                plus_di[i] = (plus_di[i-1] * 13 + plus_dm[i-1]) / 14
                minus_di[i] = (minus_di[i-1] * 13 + minus_dm[i-1]) / 14
        else:
            atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
            plus_di[i] = (plus_di[i-1] * 13 + plus_dm[i-1]) / 14
            minus_di[i] = (minus_di[i-1] * 13 + minus_dm[i-1]) / 14
    
    for i in range(14, n):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        else:
            dx[i] = 0
    
    for i in range(28, n):
        if i == 28:
            adx[i] = np.mean(dx[14:28])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i-1]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(28, 14)  # Ensure RSI and ADX ready
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Range condition: ADX < 20 indicates weak trend / ranging market
            if adx[i] < 20:
                # Long: RSI below 40 (oversold)
                if rsi[i] < 40:
                    signals[i] = 0.25
                    position = 1
                # Short: RSI above 60 (overbought)
                elif rsi[i] > 60:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (above 50) or trend strengthens
            if rsi[i] > 50 or adx[i] > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (below 50) or trend strengthens
            if rsi[i] < 50 or adx[i] > 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI40_60_MeanReversion_Range"
timeframe = "4h"
leverage = 1.0