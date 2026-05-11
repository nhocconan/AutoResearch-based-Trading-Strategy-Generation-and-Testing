#!/usr/bin/env python3
name = "6h_KAMA_RSI_Trend_Signal_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1D data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # KAMA on 1D
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    er[1:] = change[1:] / (volatility + 1e-10)
    sc = (er * 0.1 + 0.06) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # RSI on 1D
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align KAMA and RSI
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # KAMA trend on 6H
    change_6h = np.abs(np.diff(close, prepend=close[0]))
    volatility_6h = np.abs(np.diff(close))
    er_6h = np.zeros_like(close)
    er_6h[1:] = change_6h[1:] / (volatility_6h + 1e-10)
    sc_6h = (er_6h * 0.1 + 0.06) ** 2
    kama_6h = np.zeros_like(close)
    kama_6h[0] = close[0]
    for i in range(1, len(close)):
        kama_6h[i] = kama_6h[i-1] + sc_6h[i] * (close[i] - kama_6h[i-1])
    
    # Volume filter: volume > 1.2x 30-period average
    vol_ma30 = np.zeros(n)
    for i in range(n):
        if i < 30:
            vol_ma30[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(kama_6h[i]) or np.isnan(vol_ma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend direction from 1D KAMA
        trend_up = close_1d[-1] > kama_1d[-1] if len(close_1d) > 0 else False  # Simplified for alignment
        trend_down = close_1d[-1] < kama_1d[-1] if len(close_1d) > 0 else False
        
        if position == 0:
            # Long: price > 1D KAMA, RSI > 50, volume surge
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and volume[i] > 1.2 * vol_ma30[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < 1D KAMA, RSI < 50, volume surge
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and volume[i] > 1.2 * vol_ma30[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < 1D KAMA or RSI < 40
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > 1D KAMA or RSI > 60
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals