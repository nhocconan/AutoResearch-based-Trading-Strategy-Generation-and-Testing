#!/usr/bin/env python3
# Hypothesis: 1d KAMA direction with 1w EMA trend filter and volume confirmation
# Long when KAMA turns upward, price > 1w EMA20, and volume > 1.5x average
# Short when KAMA turns downward, price < 1w EMA20, and volume > 1.5x average
# Exit when KAMA reverses direction
# Combines trend-following (KAMA) with higher-timeframe trend filter (1w EMA) and volume to avoid false signals
# Designed for low-frequency, high-conviction trades on 1d timeframe
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25

name = "1d_KAMA_1wEMA20_Trend_Volume"
timeframe = "1d"
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
    
    # Calculate KAMA (10-period ER, 2 and 30 for SC)
    er_period = 10
    fast_sc = 2
    slow_sc = 30
    
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1w EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20 = np.zeros_like(close_1w)
    ema_20[0] = close_1w[0]
    alpha = 2 / (20 + 1)
    
    for i in range(1, len(close_1w)):
        ema_20[i] = ema_20[i-1] + alpha * (close_1w[i] - ema_20[i-1])
    
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(ema_20_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA upward, price > 1w EMA20, volume confirmation
            if (kama[i] > kama[i-1] and 
                close[i] > ema_20_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA downward, price < 1w EMA20, volume confirmation
            elif (kama[i] < kama[i-1] and 
                  close[i] < ema_20_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns downward
            if kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns upward
            if kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals