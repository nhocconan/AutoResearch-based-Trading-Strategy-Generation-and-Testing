# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with volume confirmation and trend filter.
- Long when price breaks above Donchian(20) high + volume > 1.5x 20-period average + price > 50-period EMA (trend)
- Short when price breaks below Donchian(20) low + volume > 1.5x 20-period average + price < 50-period EMA
- Exit on opposite Donchian break (lower for long, upper for short)
- Uses 12h EMA50 as higher timeframe trend filter (aligned) to avoid counter-trend trades
- Discrete position sizing (0.25) to minimize fee churn
- Target: 20-40 trades/year (~80-160 total over 4 years) to stay within fee limits
"""

name = "4h_Donchian20_VolumeTrend_12hEMA50"
timeframe = "4h"
leverage = 1.0

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
    
    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_12h_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    trend_up = close > ema_12h_50_aligned  # price above 12h EMA50 = uptrend
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_12h_50_aligned[i]) or np.isnan(vol_ma20[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high + uptrend + volume
            if close[i] > donch_high[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + downtrend + volume
            elif close[i] < donch_low[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian low
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian high
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals