# #!/usr/bin/env python3
# Hypothesis: 6h timeframe with 1-day RSI divergence and volume confirmation
# RSI divergence signals potential reversals, volume confirms strength
# Works in both bull and bear markets by catching exhaustion moves
# Target: 15-30 trades/year (60-120 total over 4 years)
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
    
    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    
    # Align RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume filter: volume > 2 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Price momentum: 6-period rate of change
    roc_6 = np.full(n, np.nan)
    for i in range(6, n):
        if close[i-6] != 0:
            roc_6[i] = (close[i] - close[i-6]) / close[i-6]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI (14), volume MA (20), ROC (6)
    start_idx = max(14, 20, 6)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(roc_6[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        rsi = rsi_1d_aligned[i]
        momentum = roc_6[i]
        
        # Volume filter
        vol_filter = vol_now > 2.0 * vol_avg
        
        # RSI conditions for divergence-like signals
        # Bullish: RSI oversold (<30) with positive momentum
        # Bearish: RSI overbought (>70) with negative momentum
        if position == 0:
            # Long: RSI oversold + upward momentum + volume
            if rsi < 30 and momentum > 0 and vol_filter:
                signals[i] = size
                position = 1
            # Short: RSI overbought + downward momentum + volume
            elif rsi > 70 and momentum < 0 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI overbought or momentum turns negative
            if rsi > 70 or momentum < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI oversold or momentum turns positive
            if rsi < 30 or momentum > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI_Divergence_Volume"
timeframe = "6h"
leverage = 1.0