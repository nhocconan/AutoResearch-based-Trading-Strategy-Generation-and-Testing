#!/usr/bin/env python3
name = "1d_KAMA_Trend_RSI_Threshold"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # KAMA on weekly data
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    direction = np.abs(np.diff(close_1w, k=10, prepend=close_1w[:10]))
    er = np.where(change != 0, direction / change, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Weekly RSI
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily volume spike: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA + RSI > 50 + volume spike
            if (close[i] > kama_1w_aligned[i] and 
                rsi_1w_aligned[i] > 50 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + RSI < 50 + volume spike
            elif (close[i] < kama_1w_aligned[i] and 
                  rsi_1w_aligned[i] < 50 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below KAMA
            if close[i] < kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above KAMA
            if close[i] > kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals