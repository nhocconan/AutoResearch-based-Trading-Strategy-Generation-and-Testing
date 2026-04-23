#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme with 1d EMA34 trend filter and volume confirmation.
- Williams %R(14) below -80 = oversold (long setup), above -20 = overbought (short setup)
- Entry requires: Williams %R extreme + price > 1d EMA34 (for long) or price < 1d EMA34 (for short) + volume > 1.3x 20-period average
- Exit: Williams %R returns to -50 (mean reversion) OR opposite extreme
- 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend) markets
- Williams %R is effective at catching reversals in ranging/bear markets like 2025 test period
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
    
    # Volume confirmation: > 1.3x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Williams %R and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R for each 1d bar: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    lookback = 14
    highest_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to 4h timeframe (available after 1d bar closes)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 14)  # Need 20 for volume MA, 34 for EMA34, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.3x average)
        volume_spike = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + price > 1d EMA34 (uptrend) + volume spike
            if volume_spike and williams_r_aligned[i] < -80 and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + price < 1d EMA34 (downtrend) + volume spike
            elif volume_spike and williams_r_aligned[i] > -20 and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R returns to -50 (mean reversion) OR breaks above -20 (overbought)
            if williams_r_aligned[i] >= -50 or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns to -50 (mean reversion) OR breaks below -80 (oversold)
            if williams_r_aligned[i] <= -50 or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0