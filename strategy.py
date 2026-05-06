#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week exponential moving average crossover with volume confirmation
# - Long when 1d close crosses above 1w EMA21 with volume above 20-period average
# - Short when 1d close crosses below 1w EMA21 with volume above 20-period average
# - Uses 1w EMA21 as primary trend filter (slow EMA to reduce whipsaw)
# - Volume filter ensures trades occur only during periods of conviction
# - Designed to work in both bull and bear markets by following the higher timeframe trend
# - Target: 20-60 total trades over 4 years (5-15/year) with 0.25 position sizing

name = "1d_EMA21_Crossover_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA21 calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1w EMA21
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume filter: volume above 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to allow for crossover detection
        # Skip if any critical value is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(ema_21_1w_aligned[i-1]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price crosses above EMA21 with volume confirmation
            if close[i] > ema_21_1w_aligned[i] and close[i-1] <= ema_21_1w_aligned[i-1] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below EMA21 with volume confirmation
            elif close[i] < ema_21_1w_aligned[i] and close[i-1] >= ema_21_1w_aligned[i-1] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below EMA21
            if close[i] < ema_21_1w_aligned[i] and close[i-1] >= ema_21_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above EMA21
            if close[i] > ema_21_1w_aligned[i] and close[i-1] <= ema_21_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals