#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1w EMA50 trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power turns positive (from negative/zero) with 1w EMA50 uptrend and volume > 1.5x average.
# Short when Bear Power turns negative (from positive/zero) with 1w EMA50 downtrend and volume > 1.5x average.
# Exit when power crosses back through zero (mean reversion).
# Elder Ray captures institutional buying/selling pressure; works in bull/bear via trend filter.
# Target: 50-150 total trades over 4 years = 12-37/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema13_period = 13
    ema13 = np.full(n, np.nan)
    if n >= ema13_period:
        ema13[ema13_period - 1] = np.mean(close[:ema13_period])
        for i in range(ema13_period, n):
            ema13[i] = (close[i] * (2 / (ema13_period + 1)) + 
                        ema13[i - 1] * (1 - (2 / (ema13_period + 1))))
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Previous values for crossover detection
    bull_power_prev = np.full(n, np.nan)
    bear_power_prev = np.full(n, np.nan)
    bull_power_prev[1:] = bull_power[:-1]
    bear_power_prev[1:] = bear_power[:-1]
    
    # Align 1w EMA50 to 6h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA13, EMA50(1w), and volume MA20
    start_idx = max(ema13_period, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_prev[i]) or np.isnan(bear_power_prev[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: Bull Power turns positive (from <=0) with 1w EMA50 uptrend and volume filter
            if (bull_power_prev[i] <= 0 and bull_power[i] > 0 and 
                close[i] > ema_1w_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: Bear Power turns negative (from >=0) with 1w EMA50 downtrend and volume filter
            elif (bear_power_prev[i] >= 0 and bear_power[i] < 0 and 
                  close[i] < ema_1w_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power crosses back below zero
            if bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bear Power crosses back above zero
            if bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_Energy_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0