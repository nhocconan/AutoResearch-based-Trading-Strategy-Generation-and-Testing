#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d EMA50 trend filter and volume confirmation.
- Williams %R: Measures overbought/oversold levels (-100 to 0)
- Long: Williams %R < -80 (extreme oversold) + price > 1d EMA50 (uptrend) + volume > 1.8x 20-period avg
- Short: Williams %R > -20 (extreme overbought) + price < 1d EMA50 (downtrend) + volume > 1.8x 20-period avg
- Exit: Williams %R crosses above -50 for long, below -50 for short (mean reversion completion)
- 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- Extreme readings work in both bull (oversold bounces in uptrend) and bear (overbought failures in downtrend)
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.8x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for Williams %R, 50 for 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R < -80 (extreme oversold) + price > 1d EMA50 (uptrend) + volume spike
            if volume_spike and williams_r[i] < -80 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (extreme overbought) + price < 1d EMA50 (downtrend) + volume spike
            elif volume_spike and williams_r[i] > -20 and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (mean reversion completion)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (mean reversion completion)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0