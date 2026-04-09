#!/usr/bin/env python3
# 1d_1w_ema_bounce_v1
# Hypothesis: Daily price bounces off 50-period EMA when aligned with weekly trend.
# Long when price touches EMA50 from above with weekly EMA50 uptrend and volume > 1.5x average.
# Short when price touches EMA50 from below with weekly EMA50 downtrend and volume > 1.5x average.
# Uses EMA crossovers for trend confirmation to reduce whipsaw.
# Position size fixed at 0.25 to balance risk and return.
# Target: 10-25 trades/year (40-100 total over 4 years) by requiring trend alignment and volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_bounce_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema = close_1w[49]  # Initialize with first 50-period average
        multiplier = 2 / (50 + 1)
        ema_50_1w[49] = ema
        for i in range(50, len(close_1w)):
            ema = (close_1w[i] - ema) * multiplier + ema
            ema_50_1w[i] = ema
    
    # Align weekly EMA50 to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily EMA50
    ema_50 = np.full(n, np.nan)
    if n >= 50:
        ema = close[49]  # Initialize with first 50-period average
        multiplier = 2 / (50 + 1)
        ema_50[49] = ema
        for i in range(50, n):
            ema = (close[i] - ema) * multiplier + ema
            ema_50[i] = ema
    
    # Volume confirmation: 20-period average
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
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(ema_50[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA50
            if close[i] < ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA50
            if close[i] > ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches EMA50 from above with weekly uptrend and volume confirmation
            if (close[i] >= ema_50[i] and 
                close[i-1] > ema_50[i-1] and  # Was above EMA yesterday
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and  # Weekly EMA rising
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price touches EMA50 from below with weekly downtrend and volume confirmation
            elif (close[i] <= ema_50[i] and 
                  close[i-1] < ema_50[i-1] and  # Was below EMA yesterday
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and  # Weekly EMA falling
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals