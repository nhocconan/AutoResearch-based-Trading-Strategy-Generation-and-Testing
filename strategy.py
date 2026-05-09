#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day Williams %R oversold/overbought levels combined with 12-hour EMA trend filter and volume confirmation.
# In oversold conditions (Williams %R < -80) with price above 12h EMA50 and above average volume, go long.
# In overbought conditions (Williams %R > -20) with price below 12h EMA50 and above average volume, go short.
# Williams %R identifies extreme short-term reversals, EMA50 filters for trend direction, volume confirms conviction.
# Designed to work in both bull and bear markets by capturing mean-reversion within the prevailing trend.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_WilliamsR_EMA50_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    highest_high = high_1d.rolling(window=14, min_periods=14).max()
    lowest_low = low_1d.rolling(window=14, min_periods=14).min()
    williams_r = -100 * ((highest_high - close_1d) / (highest_high - lowest_low))
    
    williams_r_values = williams_r.values
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_values)
    
    # Oversold/overbought thresholds
    oversold = williams_r_aligned < -80
    overbought = williams_r_aligned > -20
    
    # 12-hour EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Price above/below EMA50
    price_above_ema = close > ema_50
    price_below_ema = close < ema_50
    
    # Volume confirmation: above 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > avg_volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or
            np.isnan(ema_50[i]) or
            np.isnan(price_above_ema[i]) or np.isnan(price_below_ema[i]) or
            np.isnan(avg_volume[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: oversold + price above EMA50 + volume confirmation
            if oversold[i] and price_above_ema[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought + price below EMA50 + volume confirmation
            elif overbought[i] and price_below_ema[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: overbought OR price crosses below EMA50
            if overbought[i] or (not price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: oversold OR price crosses above EMA50
            if oversold[i] or (not price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals