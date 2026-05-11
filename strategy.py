#!/usr/bin/env python3
"""
4h_KAMA_Trend_12hRSI_Extreme_Confluence
Hypothesis: Use KAMA trend direction on 4h combined with extreme RSI on 12h to capture trend continuation with mean-reversion exhaustion. Designed for low trade frequency and high win rate in both bull and bear markets.
"""

name = "4h_KAMA_Trend_12hRSI_Extreme_Confluence"
timeframe = "4h"
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
    
    # Calculate KAMA on 4h
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = np.where(close > kama, 1, -1)
    
    # Get 12h data for RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_loss / (avg_gain + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_dir[i]) or 
            np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up AND RSI oversold (<30) AND volume filter
            if kama_dir[i] == 1 and rsi_12h_aligned[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down AND RSI overbought (>70) AND volume filter
            elif kama_dir[i] == -1 and rsi_12h_aligned[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA down OR RSI overbought
            if kama_dir[i] == -1 or rsi_12h_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: KAMA up OR RSI oversold
            if kama_dir[i] == 1 or rsi_12h_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals