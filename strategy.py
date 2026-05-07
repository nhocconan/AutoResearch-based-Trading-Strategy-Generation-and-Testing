#!/usr/bin/env python3
name = "4h_KAMA_TRIX_Confluence"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA and TRIX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d)).cumsum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # Calculate TRIX on 1d close (15-period EMA of EMA of EMA)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    
    # Align KAMA and TRIX to 4h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 30)
    
    for i in range(start_idx, n):
        if np.isnan(kama_1d_aligned[i]) or np.isnan(trix_1d_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA AND TRIX > 0 (bullish momentum) + volume
            if close[i] > kama_1d_aligned[i] and trix_1d_aligned[i] > 0 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA AND TRIX < 0 (bearish momentum) + volume
            elif close[i] < kama_1d_aligned[i] and trix_1d_aligned[i] < 0 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses KAMA in opposite direction
            if position == 1:
                if close[i] < kama_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals