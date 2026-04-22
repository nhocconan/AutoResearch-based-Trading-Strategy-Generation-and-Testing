#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with 1-day ATR volatility filter and volume confirmation
# Uses 1-day ATR to normalize breakout strength and filter weak breakouts in low volatility
# Volume confirmation ensures institutional participation
# Designed for 12h timeframe to reduce trade frequency and avoid fee drag
# Works in bull/bear via volatility-adjusted breakout strength

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ATR calculation and volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period ATR on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period Donchian channels on 12-hour data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(20, n):
        # Skip if data not ready or outside session
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(atr_20[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above upper Donchian with volatility filter and volume
            breakout_strength = (close[i] - high_max_20[i]) / atr_20[i]
            if (close[i] > high_max_20[i] and 
                breakout_strength > 0.3 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower Donchian with volatility filter and volume
            elif (close[i] < low_min_20[i] and 
                  (low_min_20[i] - close[i]) / atr_20[i] > 0.3 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to middle of channel or opposite breakout
            if position == 1:
                mid = (high_max_20[i] + low_min_20[i]) / 2
                if close[i] < mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                mid = (high_max_20[i] + low_min_20[i]) / 2
                if close[i] > mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_ATR_Volume_Breakout"
timeframe = "12h"
leverage = 1.0