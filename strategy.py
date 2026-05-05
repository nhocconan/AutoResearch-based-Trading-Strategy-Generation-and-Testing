#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 12h EMA50 trend filter and 1d volume spike
# Long when: Williams %R(14) crosses above -80 from below, price > 12h EMA50, and volume > 2.0x 24-period average (1d equivalent)
# Short when: Williams %R(14) crosses below -20 from above, price < 12h EMA50, and volume > 2.0x 24-period average
# Exit when: Williams %R returns to -50 level (mean reversion)
# Uses Williams %R for overextended conditions in 6h, 12h EMA for trend alignment, 1d volume for conviction
# Timeframe: 6h, HTF: 12h/1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WilliamsRExtreme_12hEMA50_1dVolumeSpike"
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
    
    # Calculate Williams %R on 6h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    if len(high) >= lookback:
        highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
        lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero (when highest_high == lowest_low)
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, np.nan)
    
    # Calculate volume confirmation on 6h using 24-period MA (equivalent to 1d lookback: 24*6h = 144h ~ 6d, but we use 24 for 1d equivalent in 6h bars)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (2.0 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 trend filter
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
            # Long conditions: Williams %R crosses above -80 from below, price > 12h EMA50, volume filter
            if (williams_r[i] > -80 and 
                williams_r[i-1] <= -80 and  # Cross above -80
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 from above, price < 12h EMA50, volume filter
            elif (williams_r[i] < -20 and 
                  williams_r[i-1] >= -20 and  # Cross below -20
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion)
            if williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion)
            if williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals