#!/usr/bin/env python3
"""
4h_RSI_Trend_Reversal_v1
Long: RSI < 30 + price above 200-day EMA (trend filter)
Short: RSI > 70 + price below 200-day EMA
Exit: RSI crosses back to 50
Uses 200-day EMA from daily timeframe as trend filter to avoid counter-trend trades.
Designed to work in both bull and bear markets by fading extremes in the direction of the higher timeframe trend.
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
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 200-day EMA from daily timeframe ===
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
            # Long: RSI < 30, price above 200-day EMA
            if (rsi[i] < 30 and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI > 70, price below 200-day EMA
            elif (rsi[i] > 70 and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI > 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Trend_Reversal_v1"
timeframe = "4h"
leverage = 1.0