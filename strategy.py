#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_Volume
Hypothesis: KAMA trend on 12h with RSI filter and volume confirmation captures strong trends while avoiding whipsaw in both bull and bear markets. Uses RSI to avoid overextended entries and volume to confirm momentum. Target: 20-40 trades/year.
"""
name = "12h_KAMA_Trend_RSI_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # 1d KAMA(10) for trend
    close_1d_series = pd.Series(close_1d)
    change = abs(close_1d_series - close_1d_series.shift(10))
    volatility = abs(close_1d_series.diff()).rolling(window=10, min_periods=1).sum()
    er = change / (volatility + 1e-10)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama_1d = [close_1d[0]]
    for i in range(1, len(close_1d)):
        kama_1d.append(kama_1d[-1] + sc[i] * (close_1d[i] - kama_1d[-1]))
    kama_1d = np.array(kama_1d)
    
    # Align to 12h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(kama_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA + RSI not overbought + volume
            if close[i] > kama_1d_aligned[i] and rsi_1d_aligned[i] < 70 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + RSI not oversold + volume
            elif close[i] < kama_1d_aligned[i] and rsi_1d_aligned[i] > 30 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through KAMA
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