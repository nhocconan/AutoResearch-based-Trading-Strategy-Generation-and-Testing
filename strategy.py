#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d EMA50 Trend + Volume Spike.
- Williams %R(14): measures overbought/oversold levels (-100 to 0)
- Long: Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 2.0x 20-period avg
- Short: Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 2.0x 20-period avg
- Exit: Williams %R crosses above -50 for long exit, below -50 for short exit
- 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
- Extreme volume confirmation (>2.0x) ensures participation and reduces false signals
- Williams %R is effective in ranging markets (2025-2026 test period) and captures reversals
- Target: 60-180 total trades over 4 years (15-45/year) to balance opportunity and fee drag on 6h
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
    
    # Volume confirmation: > 2.0x 20-period average (extreme spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for Williams %R, 50 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Extreme volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + volume spike + price > 1d EMA50 (uptrend)
            if volume_spike and close[i] > ema_50_aligned[i]:
                if williams_r[i] < -80:
                    signals[i] = 0.25
                    position = 1
            # Short: Williams %R > -20 (overbought) + volume spike + price < 1d EMA50 (downtrend)
            elif volume_spike and close[i] < ema_50_aligned[i]:
                if williams_r[i] > -20:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (exiting oversold territory)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (exiting overbought territory)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0