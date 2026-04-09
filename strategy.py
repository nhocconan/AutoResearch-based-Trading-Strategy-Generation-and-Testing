#!/usr/bin/env python3
# 1h_4d_rsi_ema_v1
# Hypothesis: On 1h timeframe, use RSI(14) and EMA(50) for mean reversion entries aligned with 4h trend.
# Long when RSI < 30 and price > 4h EMA50, short when RSI > 70 and price < 4h EMA50.
# Exit when RSI crosses back to neutral (40 for longs, 60 for shorts).
# Uses 4h EMA for trend filter and RSI for mean reversion signals.
# Position size 0.20 to limit drawdown. Target: 60-150 total trades over 4 years.
# Works in bull markets via trend-aligned mean reversion and in bear markets via oversold/overbought bounces.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_rsi_ema_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema = np.mean(close_4h[:50])
        multiplier = 2 / (50 + 1)
        ema_50_4h[49] = ema
        for i in range(50, len(close_4h)):
            ema = (close_4h[i] - ema) * multiplier + ema
            ema_50_4h[i] = ema
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Initialize first average
    if n >= 14:
        avg_gain[13] = np.mean(gain[1:15])  # Skip first element (prepended)
        avg_loss[13] = np.mean(loss[1:15])
        
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(n, np.nan)
    rsi = np.full(n, 50.0)  # Default to neutral
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 40 (mean reversion complete)
            if rsi[i] >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 60 (mean reversion complete)
            if rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: RSI oversold (<30) and price above 4h EMA50 (uptrend filter)
            if rsi[i] < 30 and close[i] > ema_50_4h_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Enter short: RSI overbought (>70) and price below 4h EMA50 (downtrend filter)
            elif rsi[i] > 70 and close[i] < ema_50_4h_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals