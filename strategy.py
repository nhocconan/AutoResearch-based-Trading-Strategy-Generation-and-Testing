#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation
# Uses 1-week high/low as trend filter and 6h Donchian(20) for entry timing
# Volume spike confirms breakout strength. Designed for low turnover (15-30 trades/year)
# Works in bull/bear by following weekly trend - avoids counter-trend losses

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly trend: price above/below weekly midpoint
    weekly_mid = (high_1w + low_1w) / 2.0
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align weekly trend to 6h timeframe
    weekly_mid_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_mid_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume spike + above weekly midpoint
            if (close[i] > highest_high[i] and vol_spike[i] and close[i] > weekly_mid_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume spike + below weekly midpoint
            elif (close[i] < lowest_low[i] and vol_spike[i] and close[i] < weekly_mid_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyTrend_Volume_Session"
timeframe = "6h"
leverage = 1.0