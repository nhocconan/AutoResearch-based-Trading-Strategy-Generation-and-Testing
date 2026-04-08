#!/usr/bin/env python3
# 4h_12h_donchian_breakout_volume_v1
# Hypothesis: 4-hour Donchian channel breakouts with 12-hour trend filter and volume confirmation
# work in both bull and bear markets by capturing breakouts with institutional participation.
# The 12-hour EMA filter ensures we only trade in the direction of the higher timeframe trend,
# reducing false signals during choppy periods. Volume confirmation adds confirmation of
# genuine market interest. Designed for moderate trade frequency (~25-40 trades/year) to
# minimize fee drag while maintaining edge.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 30-period EMA on 12h for trend filter
    close_12h = df_12h['close'].values
    ema30_12h = pd.Series(close_12h).ewm(span=30, min_periods=30, adjust=False).mean().values
    ema30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema30_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema30_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or loses 12h uptrend
            if close[i] < low_min[i] or close[i] < ema30_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or loses 12h downtrend
            if close[i] > high_max[i] or close[i] > ema30_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper band, 12h uptrend, volume confirmation
            if (close[i] > high_max[i] and 
                close[i] > ema30_12h_aligned[i] and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band, 12h downtrend, volume confirmation
            elif (close[i] < low_min[i] and 
                  close[i] < ema30_12h_aligned[i] and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals