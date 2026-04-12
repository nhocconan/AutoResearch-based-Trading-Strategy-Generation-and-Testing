#!/usr/bin/env python3
"""
4h_1d_RSI_Overbought_Oversold_Volume_Confirmation
Hypothesis: Uses daily RSI extremes combined with 4-hour volume confirmation to capture mean-reversion opportunities.
In bull markets, buys oversold dips; in bear markets, sells overbought rallies. Volume filter ensures institutional participation.
Designed for low-frequency, high-probability trades (target 20-40/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RSI_Overbought_Oversold_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY RSI CALCULATION ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily closes
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 4h VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(rsi_4h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        # Long: Daily RSI oversold (<30) with volume confirmation
        long_signal = (rsi_4h[i] < 30) and (vol_ratio[i] > 1.5)
        
        # Short: Daily RSI overbought (>70) with volume confirmation
        short_signal = (rsi_4h[i] > 70) and (vol_ratio[i] > 1.5)
        
        # Exit conditions
        # Exit long when RSI returns to neutral (>50) or reverses
        exit_long = (position == 1) and (rsi_4h[i] > 50)
        
        # Exit short when RSI returns to neutral (<50) or reverses
        exit_short = (position == -1) and (rsi_4h[i] < 50)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals