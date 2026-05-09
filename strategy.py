#!/usr/bin/env python3
# Hypothesis: 4h 12h KAMA trend with RSI confirmation and volume filter
# Long when 12h KAMA is rising, RSI(14) > 50, and volume > 1.5x average
# Short when 12h KAMA is falling, RSI(14) < 50, and volume > 1.5x average
# Exit when RSI crosses back to neutral (40-60 range) or KAMA reverses
# Uses KAMA for adaptive trend, RSI for momentum, volume for confirmation
# Target: 80-120 total trades over 4 years (20-30/year) with size 0.25

name = "4h_KAMA_RSI_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h KAMA for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # KAMA calculation
    close_12h = df_12h['close'].values
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER and KAMA calculation
    price_change = np.abs(np.diff(close_12h, n=10))  # 10-period change
    total_change = np.abs(close_12h[-1] - close_12h[0]) if len(close_12h) >= 10 else 0
    # For simplicity, use a simplified adaptive factor
    er = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if i >= 10:
            price_change = np.abs(close_12h[i] - close_12h[i-10])
            total_change = np.sum(np.abs(np.diff(close_12h[i-9:i+1])))
            er[i] = price_change / total_change if total_change != 0 else 0
    
    # Smooth er
    er = pd.Series(er).ewm(alpha=0.2).fillna(0).values  # using 0.2 as default ER smoothing
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Alternative simpler approach: use EMA as proxy for trend direction
    # But let's use actual KAMA with proper calculation
    
    # Recalculate KAMA properly
    window = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    change_10 = np.abs(np.diff(close_12h, n=window))
    volatility_sum = np.zeros_like(close_12h)
    for i in range(window, len(close_12h)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close_12h[i-window+1:i+1])))
    
    er = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if i >= window and volatility_sum[i] > 0:
            er[i] = change_10[i] / volatility_sum[i]
        else:
            er[i] = 0
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Calculate RSI(14) on 4h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for RSI and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising, RSI > 50, volume confirmation
            if (kama_aligned[i] > kama_aligned[i-1] and 
                rsi[i] > 50 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI < 50, volume confirmation
            elif (kama_aligned[i] < kama_aligned[i-1] and 
                  rsi[i] < 50 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI drops below 40 or KAMA reverses
            if (rsi[i] < 40) or (kama_aligned[i] < kama_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI rises above 60 or KAMA reverses
            if (rsi[i] > 60) or (kama_aligned[i] > kama_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals