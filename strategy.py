#!/usr/bin/env python3
"""
1d_1w_200EMA_Close_Crossover
Hypothesis: Buy when daily close crosses above weekly 200 EMA with volume confirmation; sell when crosses below. Uses weekly EMA to capture long-term trend, daily price action for entry, and volume filter to avoid false signals. Designed for low trade frequency (<30/year) to minimize fee drag in trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_200EMA_Close_Crossover"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY 200 EMA ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    if len(close_1w) >= 200:
        ema_200_1w = np.zeros_like(close_1w)
        ema_200_1w[0] = close_1w[0]
        alpha = 2.0 / (200 + 1)
        for i in range(1, len(close_1w)):
            ema_200_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_200_1w[i-1]
    else:
        ema_200_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly 200 EMA to daily timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume average (20-period for daily = ~20 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if np.isnan(ema_200_1w_aligned[i]) or vol_avg[i] == 0.0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Crossover signals
        cross_above = (close[i] > ema_200_1w_aligned[i]) and (i == 50 or close[i-1] <= ema_200_1w_aligned[i-1])
        cross_below = (close[i] < ema_200_1w_aligned[i]) and (i == 50 or close[i-1] >= ema_200_1w_aligned[i-1])
        
        if cross_above and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif cross_below and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals