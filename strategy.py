#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 12h EMA50 trend filter and volume confirmation.
- Williams %R: Measures overbought/oversold levels (-100 to 0)
- Long: Williams %R < -80 (oversold) + price > 12h EMA50 (uptrend) + volume > 1.8x 24-period avg
- Short: Williams %R > -20 (overbought) + price < 12h EMA50 (downtrend) + volume > 1.8x 24-period avg
- Exit: Williams %R crosses back above -50 (for long) or below -50 (for short) OR trend breaks
- 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- Extreme readings (> -80 or < -20) provide high-probability mean reversion spots in ranging markets
- Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
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
    
    # Volume confirmation: > 1.8x 24-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA50 for trend filter (using 12h data)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period) using 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 14, 50)  # Need 24 for volume MA, 14 for Williams %R, 50 for 12h EMA50
    
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
            # Long: Williams %R < -80 (oversold) + price > 12h EMA50 (uptrend) + volume spike
            if volume_spike and williams_r[i] < -80 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + price < 12h EMA50 (downtrend) + volume spike
            elif volume_spike and williams_r[i] > -20 and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -50 (momentum returning) OR price < 12h EMA50 (trend break)
            if williams_r[i] > -50 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -50 (momentum returning) OR price > 12h EMA50 (trend break)
            if williams_r[i] < -50 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0