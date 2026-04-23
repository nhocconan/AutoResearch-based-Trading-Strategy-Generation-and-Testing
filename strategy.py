#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Donchian breakout: price > 20-day high (long) or < 20-day low (short)
- Trend filter: price > 1w EMA50 for longs, price < 1w EMA50 for shorts
- Volume confirmation: volume > 1.5x 20-day average
- Exit: opposite Donchian breakout OR price crosses 1w EMA50
- Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
- Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
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
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA50 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > donchian_high[i-1]  # Price above prior 20-day high
        breakout_short = close[i] < donchian_low[i-1]  # Price below prior 20-day low
        
        # Trend filter conditions
        trend_long = close[i] > ema_50_1w_aligned[i]   # Price above 1w EMA50
        trend_short = close[i] < ema_50_1w_aligned[i]  # Price below 1w EMA50
        
        if position == 0:
            # Long: Donchian breakout above + uptrend + volume confirmation
            if breakout_long and trend_long and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below + downtrend + volume confirmation
            elif breakout_short and trend_short and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown OR price < 1w EMA50 (trend flip)
            if close[i] < donchian_low[i-1] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout OR price > 1w EMA50 (trend flip)
            if close[i] > donchian_high[i-1] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0