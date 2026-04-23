#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d EMA50 trend + volume confirmation.
- Williams %R(14): extreme oversold < -80 for long, extreme overbought > -20 for short
- Trend filter: price > 1d EMA50 for longs, price < 1d EMA50 for shorts
- Volume confirmation: volume > 2.0x 24-period average to avoid false breakouts
- Exit: Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Works in bull (buying extreme dips in uptrend) and bear (selling extreme rallies in downtrend)
- Discrete position sizing: ±0.25 to minimize fee churn
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
    
    # Volume confirmation: > 2.0x 24-period average (strong spike filter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 14)  # Need 24 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
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
            # Long exit: Williams %R crosses above -50 (momentum fading)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (momentum fading)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0