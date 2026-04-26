#!/usr/bin/env python3
"""
1h_HTF_Trend_Filter_With_Volume_Confirmation_v1
Hypothesis: Trade with 4h and 1d trend alignment using EMA crossovers, confirmed by volume spikes on 1h timeframe.
- Uses 4h EMA(21) and 1d EMA(50) for trend direction filter (requires both to agree)
- Enters on 1h breakouts of 20-period Donchian channels when volume > 2x 20-period average
- Exits when trend disagrees or price reaches opposite Donchian boundary
- Designed for 1h timeframe targeting 60-150 total trades over 4 years (15-37/year)
- Works in bull/bear markets by requiring HTF trend alignment, reducing false breakouts
- Volume confirmation ensures momentum behind moves, reducing whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for indicators
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA21 for short-term trend
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Calculate 1d EMA50 for long-term trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1h volume spike filter
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian/volume, 21 for 4h EMA, 50 for 1d EMA)
    start_idx = max(20, 21, 50)
    
    for i in range(start_idx, n):
        # Skip if any HTF data not ready
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Trend filters: both 4h and 1d must agree
        trend_up = close[i] > ema21_4h_aligned[i] and close[i] > ema50_1d_aligned[i]
        trend_down = close[i] < ema21_4h_aligned[i] and close[i] < ema50_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_20[i]
        breakout_down = close[i] < lowest_20[i]
        
        if position == 0:
            # Long: bullish alignment + breakout + volume spike
            if trend_up and breakout_up and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: bearish alignment + breakout + volume spike
            elif trend_down and breakout_down and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: trend disagrees OR price reaches opposite Donchian boundary
            if not trend_up or close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: trend disagrees OR price reaches opposite Donchian boundary
            if not trend_down or close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_HTF_Trend_Filter_With_Volume_Confirmation_v1"
timeframe = "1h"
leverage = 1.0