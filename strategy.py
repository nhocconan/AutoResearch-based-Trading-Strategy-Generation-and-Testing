#!/usr/bin/env python3
# 12h_ema_crossover_volume_v1
# Hypothesis: Uses 12h EMA crossover for trend direction and volume confirmation to reduce false signals.
# Long when EMA25 crosses above EMA50 with volume > 1.5x average; short when EMA25 crosses below EMA50 with volume > 1.5x average.
# Exit when opposite crossover occurs. Uses 1d trend filter to avoid counter-trend trades in strong trends.
# Designed for low trade frequency (10-30/year) to minimize fee drag on 12h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ema_crossover_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h EMA25 and EMA50 for crossover signals
    ema_fast = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_slow = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1d data for trend filter (close vs SMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(sma50_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: EMA25 crosses below EMA50
            if ema_fast[i] < ema_slow[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA25 crosses above EMA50
            if ema_fast[i] > ema_slow[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA25 crosses above EMA50, volume surge, and price above 1d SMA50 (uptrend filter)
            if (ema_fast[i] > ema_slow[i] and 
                ema_fast[i-1] <= ema_slow[i-1] and  # crossover just happened
                vol_surge[i] and
                close[i] > sma50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: EMA25 crosses below EMA50, volume surge, and price below 1d SMA50 (downtrend filter)
            elif (ema_fast[i] < ema_slow[i] and 
                  ema_fast[i-1] >= ema_slow[i-1] and  # crossover just happened
                  vol_surge[i] and
                  close[i] < sma50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals