#!/usr/bin/env python3
name = "12h_1d_1w_4H_Momentum_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for market structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 4h data for momentum confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.zeros(len(close_1d))
    atr_1d[0] = np.nan
    for i in range(1, len(tr)):
        if i < 14:
            atr_1d[i] = np.mean(tr[:i+1])
        else:
            atr_1d[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 4h RSI for momentum
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(len(close_4h))
    avg_loss = np.zeros(len(close_4h))
    avg_gain[0] = np.nan
    avg_loss[0] = np.nan
    
    for i in range(1, len(gain)):
        if i < 14:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(len(close_4h))
    rsi_4h = np.zeros(len(close_4h))
    for i in range(14, len(close_4h)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi_4h[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_4h[i] = 100
    
    # Align indicators to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 12h price momentum (rate of change over 3 periods)
    roc = np.zeros(n)
    for i in range(3, n):
        if close[i-3] != 0:
            roc[i] = (close[i] - close[i-3]) / close[i-3] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(rsi_4h_aligned[i]) or
            np.isnan(roc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: positive momentum + oversold RSI + volatility filter
            if (roc[i] > 0.5 and 
                rsi_4h_aligned[i] < 35 and 
                volume[i] > np.mean(np.maximum(1, volume[max(0, i-20):i]))):
                signals[i] = 0.25
                position = 1
            # Short: negative momentum + overbought RSI + volatility filter
            elif (roc[i] < -0.5 and 
                  rsi_4h_aligned[i] > 65 and 
                  volume[i] > np.mean(np.maximum(1, volume[max(0, i-20):i]))):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: momentum turns negative or RSI overbought
            if (roc[i] < -0.2 or rsi_4h_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: momentum turns positive or RSI oversold
            if (roc[i] > 0.2 or rsi_4h_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals