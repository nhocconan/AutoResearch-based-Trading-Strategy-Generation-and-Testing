#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d EMA34 slope > 0 (uptrend) AND volume > 1.5x 20-period average.
# Short when Bear Power > 0 AND Bull Power < 0 AND 1d EMA34 slope < 0 (downtrend) AND volume > 1.5x 20-period average.
# Exit when Bull Power <= 0 (for long) or Bear Power <= 0 (for short).
# Uses Elder Ray to measure bull/bear strength relative to EMA13, avoiding weak trends.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

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
    
    # 6h EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 6h Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate EMA34 slope (1-bar change) to determine trend direction
    ema34_slope = np.zeros_like(ema34_1d)
    ema34_slope[1:] = ema34_1d[1:] - ema34_1d[:-1]  # Positive = uptrend, negative = downtrend
    
    # Align 1d EMA34 slope to 6h timeframe
    ema34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema34_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0, 1d EMA34 rising, volume spike
            long_cond = (bull_power[i] > 0) and (bear_power[i] < 0) and (ema34_slope_aligned[i] > 0) and volume_filter[i]
            # Short conditions: Bear Power > 0, Bull Power < 0, 1d EMA34 falling, volume spike
            short_cond = (bear_power[i] > 0) and (bull_power[i] < 0) and (ema34_slope_aligned[i] < 0) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 (weakening bull strength)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 (weakening bear strength)
            if bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals