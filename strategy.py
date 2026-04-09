#!/usr/bin/env python3
# 1d_1w_ema_bounce_v2
# Hypothesis: Daily price bounce from weekly EMA200 with volume confirmation.
# Long when price touches weekly EMA200 from below with volume > 1.5x average.
# Short when price touches weekly EMA200 from above with volume > 1.5x average.
# Exit when price crosses weekly EMA200 in opposite direction.
# Uses weekly EMA200 for trend support/resistance and volume filter to avoid false signals.
# Position size fixed at 0.25 to limit drawdown. Target: 30-100 total trades over 4 years (7-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_bounce_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200
    close_1w = df_1w['close'].values
    ema_200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema = close_1w[199]  # Initialize with first 200-period average
        multiplier = 2 / (200 + 1)
        ema_200_1w[199] = ema
        for i in range(200, len(close_1w)):
            ema = (close_1w[i] - ema) * multiplier + ema
            ema_200_1w[i] = ema
    
    # Align weekly EMA200 to daily timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly EMA200
            if close[i] < ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly EMA200
            if close[i] > ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches weekly EMA200 from below with volume filter
            if (low[i] <= ema_200_1w_aligned[i] * 1.001 and  # Allow small tolerance
                close[i] > ema_200_1w_aligned[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price touches weekly EMA200 from above with volume filter
            elif (high[i] >= ema_200_1w_aligned[i] * 0.999 and  # Allow small tolerance
                  close[i] < ema_200_1w_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals