#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Donchian(20) upper band AND 1d EMA34 rising AND volume > 1.5x 20-period average volume
- Short when price breaks below Donchian(20) lower band AND 1d EMA34 falling AND volume > 1.5x 20-period average volume
- Exit on opposite Donchian breakout or when volume drops below average (to avoid chop)
- Fixed position size 0.25 to control risk and minimize fee churn
- Uses 4h primary with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Donchian provides objective breakout levels; 1d EMA34 filters regime; volume spike confirms institutional participation
- Designed to work in both bull (trend following) and bear (mean reversion via regime filter) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 20-period average volume for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Determine trend: rising if current > previous, falling if current < previous
    ema_rising = np.zeros_like(ema_34_1d_aligned, dtype=bool)
    ema_falling = np.zeros_like(ema_34_1d_aligned, dtype=bool)
    ema_rising[1:] = ema_34_1d_aligned[1:] > ema_34_1d_aligned[:-1]
    ema_falling[1:] = ema_34_1d_aligned[1:] < ema_34_1d_aligned[:-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper band AND rising EMA34 AND volume spike
            if close[i] > upper[i] and ema_rising[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band AND falling EMA34 AND volume spike
            elif close[i] < lower[i] and ema_falling[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below lower band OR volume drops below average (chop filter)
            if close[i] < lower[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above upper band OR volume drops below average (chop filter)
            if close[i] > upper[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0