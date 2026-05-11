#!/usr/bin/env python3
name = "6h_ADX_Trend_EMA_200_RSI_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA200 and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # EMA200 on daily
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # RSI(14) on daily
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # neutral when undefined
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 6h data for ADX
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # ADX(14) calculation
    plus_dm = np.zeros(len(high_6h))
    minus_dm = np.zeros(len(high_6h))
    tr = np.zeros(len(high_6h))
    
    for i in range(1, len(high_6h)):
        high_diff = high_6h[i] - high_6h[i-1]
        low_diff = low_6h[i-1] - low_6h[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(high_6h[i] - low_6h[i], abs(high_6h[i] - close_6h[i-1]), abs(low_6h[i] - close_6h[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr_6h = np.zeros(len(high_6h))
    plus_di_6h = np.zeros(len(high_6h))
    minus_di_6h = np.zeros(len(high_6h))
    
    # Initial values
    atr_6h[0] = tr[0]
    plus_di_6h[0] = 0
    minus_di_6h[0] = 0
    
    for i in range(1, len(high_6h)):
        atr_6h[i] = (atr_6h[i-1] * 13 + tr[i]) / 14
        plus_di_6h[i] = 100 * (plus_dm[i] / atr_6h[i]) if atr_6h[i] != 0 else 0
        minus_di_6h[i] = 100 * (minus_dm[i] / atr_6h[i]) if atr_6h[i] != 0 else 0
    
    dx = np.zeros(len(high_6h))
    for i in range(len(high_6h)):
        di_sum = plus_di_6h[i] + minus_di_6h[i]
        dx[i] = 100 * abs(plus_di_6h[i] - minus_di_6h[i]) / di_sum if di_sum != 0 else 0
    
    adx_6h = np.zeros(len(high_6h))
    adx_6h[0] = dx[0]
    for i in range(1, len(high_6h)):
        adx_6h[i] = (adx_6h[i-1] * 13 + dx[i]) / 14
    
    adx_6h_aligned = align_htf_to_ltf(prices, df_6h, adx_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # enough for EMA200 and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(adx_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above EMA200, RSI > 50, ADX > 25
            if (close[i] > ema200_1d_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                adx_6h_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA200, RSI < 50, ADX > 25
            elif (close[i] < ema200_1d_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  adx_6h_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below EMA200 OR RSI < 40
            if close[i] < ema200_1d_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above EMA200 OR RSI > 60
            if close[i] > ema200_1d_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals