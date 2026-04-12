#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1w_kama_rsi_v1
# Uses weekly KAMA direction (trend) with RSI(14) for entry timing on 12h.
# In bull markets: KAMA up + RSI crosses above 50 → long.
# In bear markets: KAMA down + RSI crosses below 50 → short.
# Volume filter ensures participation. Target: 15-30 trades/year per symbol.

name = "12h_1w_kama_rsi_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for KAMA (trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on weekly close
    close_1w = df_1w['close'].values
    er = np.zeros_like(close_1w)
    for i in range(1, len(close_1w)):
        change = abs(close_1w[i] - close_1w[i-1])
        volatility = np.sum(np.abs(np.diff(close_1w[max(0, i-9):i+1]))) if i >= 1 else 0
        er[i] = change / volatility if volatility != 0 else 0
    
    sc = (er * 0.6 + 0.06) ** 2  # smoothing constant
    kama = np.full_like(close_1w, np.nan)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to 12h (1-week delay for completion)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # RSI on 12h close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if KAMA not ready
        if np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_filter[i]):
            signals[i] = 0.0
            continue
        
        # Check volume filter
        if not vol_filter[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # KAMA trend direction: up if current > previous, down if current < previous
        if i > 30:
            kama_up = kama_aligned[i] > kama_aligned[i-1]
            kama_down = kama_aligned[i] < kama_aligned[i-1]
        else:
            kama_up = False
            kama_down = False
        
        # Long: KAMA up + RSI crosses above 50
        if kama_up and rsi[i] > 50 and rsi[i-1] <= 50 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: KAMA down + RSI crosses below 50
        elif kama_down and rsi[i] < 50 and rsi[i-1] >= 50 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite RSI cross
        elif rsi[i] < 50 and rsi[i-1] >= 50 and position == 1:
            position = 0
            signals[i] = 0.0
        elif rsi[i] > 50 and rsi[i-1] <= 50 and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals