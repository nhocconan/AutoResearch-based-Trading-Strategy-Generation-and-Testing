#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Donchian(20) high AND 12h EMA50 is rising AND volume > 2.0 * 20-period average
- Short when price breaks below Donchian(20) low AND 12h EMA50 is falling AND volume > 2.0 * 20-period average
- Exit when price touches Donchian(20) midpoint OR volume drops below average
- Uses 4h primary with 12h HTF for trend filter to avoid counter-trend trades
- Donchian channels provide clear structure; EMA50 filters trend direction; volume confirms breakout conviction
- Designed to work in both bull (breakouts with volume) and bear (breakdowns with volume) markets
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # EMA50 slope: rising if current > previous, falling if current < previous
    ema_50_rising = np.zeros_like(ema_50_12h_aligned, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_12h_aligned, dtype=bool)
    ema_50_rising[1:] = ema_50_12h_aligned[1:] > ema_50_12h_aligned[:-1]
    ema_50_falling[1:] = ema_50_12h_aligned[1:] < ema_50_12h_aligned[:-1]
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1  # Need Donchian20, EMA50 data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND EMA50 rising AND volume spike
            if close[i] > donchian_high[i] and ema_50_rising[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND EMA50 falling AND volume spike
            elif close[i] < donchian_low[i] and ema_50_falling[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches Donchian midpoint OR volume drops below average
            if close[i] >= donchian_mid[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches Donchian midpoint OR volume drops below average
            if close[i] <= donchian_mid[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0