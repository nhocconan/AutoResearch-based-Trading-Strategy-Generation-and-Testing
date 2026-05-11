#!/usr/bin/env python3
name = "12h_KAMA_RSI_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1D data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1D KAMA - Trend direction
    price_1d = df_1d['close'].values
    change = np.abs(np.diff(price_1d, prepend=price_1d[0]))
    vol = np.abs(np.diff(price_1d))
    er = np.divide(change, vol, out=np.zeros_like(change), where=vol!=0)
    er = np.clip(er, 0, 1)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # k=2, fast=2, slow=30
    kama_1d = np.zeros_like(price_1d)
    kama_1d[0] = price_1d[0]
    for i in range(1, len(price_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (price_1d[i] - kama_1d[i-1])
    kama_dir = kama_1d > np.roll(kama_1d, 1)
    kama_dir[0] = False
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    
    # 1D RSI(14)
    delta = np.diff(price_1d, prepend=price_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:14] = np.nan
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # 12h Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, Volume surge
            if kama_dir_aligned[i] and rsi_aligned[i] > 50 and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, Volume surge
            elif not kama_dir_aligned[i] and rsi_aligned[i] < 50 and volume[i] > 1.5 * vol_ma20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA down OR RSI < 40
            if not kama_dir_aligned[i] or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA up OR RSI > 60
            if kama_dir_aligned[i] or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals