#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme + 1d EMA34 Trend + Volume Spike.
- Williams %R(14) identifies overbought/oversold extremes (< -80 for long, > -20 for short).
- 1d EMA34 provides higher-timeframe trend filter to align with intermediate momentum.
- Volume spike (>2x 24-period average) confirms extreme reversal validity.
- Discrete position sizing (0.25) minimizes fee churn.
- Works in bull/bear markets via 1d trend filter and mean-reversion from extremes.
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 12h timeframe
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    rr_diff = highest_high - lowest_low
    williams_r = np.where(rr_diff != 0, ((highest_high - close) / rr_diff) * -100, -50)
    
    # Volume confirmation: > 2.0x 24-period average volume (12h * 2 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 24, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with volume spike and above 1d EMA34 (bullish higher-timeframe trend)
            if williams_r[i] < -80 and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with volume spike and below 1d EMA34 (bearish higher-timeframe trend)
            elif williams_r[i] > -20 and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -20 (overbought) OR below 1d EMA34 (trend change)
            if williams_r[i] > -20 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -80 (oversold) OR above 1d EMA34 (trend change)
            if williams_r[i] < -80 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0