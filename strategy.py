# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h EMA filter and volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA20 AND volume > 1.5x average
# Short when Bear Power < 0 AND Bull Power < 0 AND price < 12h EMA20 AND volume > 1.5x average
# Exit when Bull Power and Bear Power have same sign (both positive or both negative)
# Uses 6h timeframe for balanced trade frequency, Elder Ray for bull/bear strength,
# 12h EMA for trend filter, volume for confirmation. Works in bull (strong bull power)
# and bear (strong bear power) markets by requiring clear bull/bear divergence.

name = "6h_elder_ray_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12h EMA20 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema20_12h = pd.Series(df_12h['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema20_12h_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exit: bull and bear power same sign (no divergence)
        if position == 1:  # long position
            if bull_power[i] <= 0 and bear_power[i] >= 0:  # both non-negative or both non-positive
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bull_power[i] >= 0 and bear_power[i] <= 0:  # both non-negative or both non-positive
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA20 AND volume confirmation
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema20_12h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power < 0 AND price < 12h EMA20 AND volume confirmation
            elif (bear_power[i] < 0 and bull_power[i] < 0 and 
                  close[i] < ema20_12h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals