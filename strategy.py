#!/usr/bin/env python3
"""
6h_RSI20_TrendFilter_V1
RSI(20) on 6h with 1d trend filter: price above/below 1d EMA200.
Long when RSI crosses above 30 (from oversold) and price > 1d EMA200.
Short when RSI crosses below 70 (from overbought) and price < 1d EMA200.
Exit when RSI crosses 50 (mean reversion) or trend fails.
Designed to capture mean reversion in trending markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === RSI(20) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=20, min_periods=20).mean().values
    avg_loss = pd.Series(loss).rolling(window=20, min_periods=20).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d EMA200 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI crosses above 30 and price above 1d EMA200
            if (rsi[i] > 30 and rsi[i-1] <= 30 and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI crosses below 70 and price below 1d EMA200
            elif (rsi[i] < 70 and rsi[i-1] >= 70 and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses below 50 OR price below 1d EMA200
            if (rsi[i] < 50 and rsi[i-1] >= 50 or 
                close[i] < ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses above 50 OR price above 1d EMA200
            if (rsi[i] > 50 and rsi[i-1] <= 50 or 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI20_TrendFilter_V1"
timeframe = "6h"
leverage = 1.0