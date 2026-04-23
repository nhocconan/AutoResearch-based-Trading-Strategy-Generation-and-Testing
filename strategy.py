#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
- Long: Close > Donchian Upper(20) + price > 1w EMA34 (uptrend) + volume > 2.0x 20-period average
- Short: Close < Donchian Lower(20) + price < 1w EMA34 (downtrend) + volume > 2.0x 20-period average
- Exit: Opposite Donchian breakout (close < Donchian Middle for longs, close > Donchian Middle for shorts)
- Uses Donchian channels for breakout structure, 1w EMA34 for major trend filter
- Volume confirmation ensures institutional participation (tighter than typical 1.5x)
- Discrete position sizing (0.25) to minimize fee churn
- Target: 50-100 total trades over 4 years (12-25/year) to avoid fee drag
- Works in both bull and bear markets by capturing strong directional moves with trend alignment
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
    
    # Calculate Donchian channels (20-period)
    # Upper: highest high over 20 periods
    # Lower: lowest low over 20 periods
    # Middle: average of upper and lower
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: > 2.0x 20-period average (tighter filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 20)  # Donchian needs 20, EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1w EMA34
        uptrend = close[i] > ema34_1w_aligned[i]
        downtrend = close[i] < ema34_1w_aligned[i]
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: Close > Donchian Upper(20) + uptrend + volume spike
        # Short: Close < Donchian Lower(20) + downtrend + volume spike
        long_signal = (close[i] > donchian_upper[i] and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < donchian_lower[i] and 
                       downtrend and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Opposite Donchian breakout (return to middle)
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to or below Donchian Middle
                if close[i] <= donchian_middle[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price returns to or above Donchian Middle
                if close[i] >= donchian_middle[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0