#!/usr/bin/env python3
# 4h_KAMA_RSI_Trend_Filter
# Hypothesis: Use 1d KAMA trend for direction, 4h RSI for entry timing, and volatility filter to avoid whipsaws.
# Long when 1d KAMA rising and 4h RSI < 40 with volatility above median; short when 1d KAMA falling and 4h RSI > 60.
# Designed for low trade frequency (15-30/year) to avoid fee drift, works in bull/bear via trend filter.

name = "4h_KAMA_RSI_Trend_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d data for KAMA trend and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d KAMA (using close)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_up = kama > np.roll(kama, 1)
    kama_down = kama < np.roll(kama, 1)
    kama_up[0] = False
    kama_down[0] = False
    
    # 1d volatility filter: ATR(14) normalized by price
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]), np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[0], tr1])
    atr14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    vol_filter = atr14_1d / close_1d
    vol_median = pd.Series(vol_filter).rolling(window=50, min_periods=50).median().values
    vol_ok = vol_filter >= vol_median  # Trade when volatility is at least median
    
    # Align 1d indicators to 4h
    kama_up_aligned = align_htf_to_ltf(prices, df_1d, kama_up.astype(float))
    kama_down_aligned = align_htf_to_ltf(prices, df_1d, kama_down.astype(float))
    vol_ok_aligned = align_htf_to_ltf(prices, df_1d, vol_ok.astype(float))
    
    # 4h RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi_oversold = rsi < 40
    rsi_overbought = rsi > 60
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama_up_aligned[i]) or np.isnan(kama_down_aligned[i]) or
            np.isnan(vol_ok_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d KAMA up + volatility OK + RSI oversold
            if (kama_up_aligned[i] > 0.5 and 
                vol_ok_aligned[i] > 0.5 and
                rsi_oversold[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1d KAMA down + volatility OK + RSI overbought
            elif (kama_down_aligned[i] > 0.5 and 
                  vol_ok_aligned[i] > 0.5 and
                  rsi_overbought[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA trend fails or volatility too low
            if (kama_up_aligned[i] < 0.5 or 
                vol_ok_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA trend fails or volatility too low
            if (kama_down_aligned[i] < 0.5 or 
                vol_ok_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals