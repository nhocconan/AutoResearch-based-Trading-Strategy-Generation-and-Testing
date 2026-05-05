#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 12h trend filter and volume confirmation
# Long when Williams %R(14) < -80 (oversold) AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when Williams %R(14) > -20 (overbought) AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short)
# Uses 6h primary timeframe for Williams %R calculation and volume confirmation
# 12h HTF for EMA50 trend filter to avoid counter-trend trades
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_WilliamsR_Extreme_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R on 6h
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, -50)
    
    # Calculate volume spike filter on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold AND above 12h EMA50 AND volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought AND below 12h EMA50 AND volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (recovery from oversold)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (recovery from overbought)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals