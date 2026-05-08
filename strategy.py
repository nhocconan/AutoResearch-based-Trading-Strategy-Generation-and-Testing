#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Long when Bull Power > 0, Bear Power < 0, price > EMA34(1d), and volume > 1.5x 20-period average.
# Short when Bull Power < 0, Bear Power > 0, price < EMA34(1d), and volume > 1.5x 20-period average.
# Exit when Bull Power and Bear Power cross (Bull Power < Bear Power for long exit, Bull Power > Bear Power for short exit).
# Elder Ray measures bull/bear strength relative to EMA13. EMA34 filters trend direction.
# Volume confirms conviction. Target: 60-120 total trades over 4 years (15-30/year).

name = "6h_ElderRay_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for EMA34 and Elder Ray components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0, price > EMA34, volume filter
            long_cond = (bull_power_aligned[i] > 0) and (bear_power_aligned[i] < 0) and (close[i] > ema_34_aligned[i]) and volume_filter[i]
            # Short conditions: Bull Power < 0, Bear Power > 0, price < EMA34, volume filter
            short_cond = (bull_power_aligned[i] < 0) and (bear_power_aligned[i] > 0) and (close[i] < ema_34_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power crosses below Bear Power
            if bull_power_aligned[i] < bear_power_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power crosses above Bear Power
            if bull_power_aligned[i] > bear_power_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals