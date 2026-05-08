#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter, volume confirmation, and session filter (08-20 UTC).
# Uses 4h EMA(50) for trend direction and 1h Donchian channels (20-period) for breakout signals.
# Volume spike (>1.5x 20-period average) confirms breakout strength.
# Designed for low trade frequency in both bull and bear markets by requiring multiple confluences.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.

name = "1h_Donchian_4hTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_4h_val = ema50_4h_aligned[i]
        upper_channel = high_roll[i]
        lower_channel = low_roll[i]
        vol_spike = volume_spike[i]
        in_session = session_filter[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel + uptrend + volume spike + session
            if (close[i] > upper_channel and 
                close[i] > ema50_4h_val and 
                vol_spike and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below lower channel + downtrend + volume spike + session
            elif (close[i] < lower_channel and 
                  close[i] < ema50_4h_val and 
                  vol_spike and 
                  in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower channel OR trend reverses
            if (close[i] < lower_channel or close[i] < ema50_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above upper channel OR trend reverses
            if (close[i] > upper_channel or close[i] > ema50_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals