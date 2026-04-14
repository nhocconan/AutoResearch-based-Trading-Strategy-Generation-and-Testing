#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with 1-week EMA(21) trend filter and volume confirmation.
# The 1-week EMA(21) filters for the dominant weekly trend, avoiding counter-trend trades.
# The 1-day Donchian(20) breakout captures momentum in the direction of the weekly trend.
# Volume > 1.5x the 20-day average confirms institutional participation.
# Exit when price returns to the weekly EMA(21) or breaks the opposite Donchian band.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the weekly trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1-week EMA(21) for trend filter
    ema_len = 21
    if len(df_1w) < ema_len:
        return np.zeros(n)
    
    ema_1w = pd.Series(df_1w['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Donchian channel (20 periods) on 1-day
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.5x average volume (20-day)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1-week EMA21
        above_ema = close[i] > ema_1w_aligned[i]
        below_ema = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + above 1-week EMA + volume
            if (close[i] > dc_upper[i] and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + below 1-week EMA + volume
            elif (close[i] < dc_lower[i] and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 1-week EMA or breaks below Donchian lower
            if close[i] < ema_1w_aligned[i] or close[i] < dc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 1-week EMA or breaks above Donchian upper
            if close[i] > ema_1w_aligned[i] or close[i] > dc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_EMA21_Donchian_Volume_v1"
timeframe = "1d"
leverage = 1.0