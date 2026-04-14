#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with weekly MA(50) trend filter and volume confirmation.
# The weekly MA(50) adapts to both bull and bear markets, ensuring trades follow the dominant trend.
# The Donchian(20) breakout captures momentum in the direction of the weekly trend.
# Volume > 1.5x the 20-period average confirms institutional participation and reduces false breakouts.
# Exit occurs when price returns to the weekly MA(50) or breaks the opposite Donchian band.
# This combination aims for 10-25 trades per year per symbol (40-100 total over 4 years), staying within the optimal range to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w MA(50) for trend filter
    ma_len = 50
    if len(df_1w) < ma_len:
        return np.zeros(n)
    
    ma_1w = pd.Series(df_1w['close']).rolling(window=ma_len, min_periods=ma_len).mean().values
    ma_1w_aligned = align_htf_to_ltf(prices, df_1w, ma_1w)
    
    # Donchian channel (20 periods) on 1d
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
            np.isnan(ma_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1w MA50
        above_ma = close[i] > ma_1w_aligned[i]
        below_ma = close[i] < ma_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + above 1w MA + volume
            if (close[i] > dc_upper[i] and 
                above_ma and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + below 1w MA + volume
            elif (close[i] < dc_lower[i] and 
                  below_ma and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 1w MA or breaks below Donchian lower
            if close[i] < ma_1w_aligned[i] or close[i] < dc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 1w MA or breaks above Donchian upper
            if close[i] > ma_1w_aligned[i] or close[i] > dc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_MA50_Donchian_Volume_v1"
timeframe = "1d"
leverage = 1.0