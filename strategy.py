#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Elder Ray (Bull/Bear Power) and 1w trend filter.
# Long: Bull Power > 0 and weekly EMA(20) rising (trend up).
# Short: Bear Power < 0 and weekly EMA(20) falling (trend down).
# Uses 1d Elder Ray for momentum, 1w EMA for trend filter. Avoids counter-trend trades.
# Time filter: none (use all hours).
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d data for Elder Ray (Bull/Bear Power)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # 1w data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        ema20w = ema20_1w_aligned[i]
        
        # Need previous EMA to check if rising/falling
        if i == 20:
            prev_ema20w = ema20w
        else:
            prev_ema20w = ema20_1w_aligned[i-1]
        
        ema_rising = ema20w > prev_ema20w
        ema_falling = ema20w < prev_ema20w
        
        if position == 0:
            # Long: Bull Power > 0 and weekly EMA rising
            if (bull > 0 and ema_rising):
                position = 1
                signals[i] = position_size
            # Short: Bear Power < 0 and weekly EMA falling
            elif (bear < 0 and ema_falling):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power <= 0 (momentum lost)
            if bull <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power >= 0 (momentum lost)
            if bear >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_1w_Elder_Ray_Trend_Filter"
timeframe = "6h"
leverage = 1.0