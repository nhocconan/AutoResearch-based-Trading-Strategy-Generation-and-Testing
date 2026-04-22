#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with daily pivot direction and volume confirmation
# Uses 6h Donchian(20) breakouts confirmed by 1d pivot direction (bullish if close > pivot)
# Requires volume spike to confirm breakout strength.
# Works in bull markets via upward breakouts, in bear via downward breakouts.
# Target: 15-25 trades/year per symbol (60-100 total) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point (standard: (H+L+C)/3)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC (aligns with major market sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align daily pivot to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(20, n):
        # Skip if data not ready or outside session
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + price above pivot + volume spike
            if (high[i] > high_max[i-1] and close[i] > pivot_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + price below pivot + volume spike
            elif (low[i] < low_min[i-1] and close[i] < pivot_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if low[i] < low_min[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if high[i] > high_max[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Pivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0