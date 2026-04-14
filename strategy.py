#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Uses 1d EMA50 as trend filter - only take long when above, short when below
# Donchian(20) breakout captures momentum, volume > 1.5x average confirms
# Exit when price touches opposite Donchian band or trend reverses
# Designed for fewer trades (target 20-30/year) to avoid fee drag
# Works in bull/bear as trend filter adapts

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channel (20 periods) on 4h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.5x average volume
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
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price vs 1d EMA50
        above_ema50 = close[i] > ema50_1d_aligned[i]
        below_ema50 = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + above 1d EMA50 + volume
            if (close[i] > dc_upper[i] and 
                above_ema50 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + below 1d EMA50 + volume
            elif (close[i] < dc_lower[i] and 
                  below_ema50 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches lower Donchian or trend turns bearish
            if close[i] < dc_lower[i] or not above_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price touches upper Donchian or trend turns bullish
            if close[i] > dc_upper[i] or not below_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_EMA50_Donchian_Volume_v1"
timeframe = "4h"
leverage = 1.0