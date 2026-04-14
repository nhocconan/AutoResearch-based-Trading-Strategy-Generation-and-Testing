#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day RSI(14) trend filter and volume confirmation.
# The 1-day RSI(14) > 50 indicates bullish bias, < 50 indicates bearish bias, ensuring trades follow the dominant trend.
# The Donchian(20) breakout captures momentum in the direction of the 1-day trend.
# Volume > 1.5x the 20-period average confirms institutional participation and reduces false breakouts.
# Exit occurs when price returns to the 1-day RSI(50) level or breaks the opposite Donchian band.
# This combination aims for 20-40 trades per year per symbol (80-160 total over 4 years), staying within the optimal range to minimize fee drift.

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
    
    # 1d RSI(14) for trend filter
    rsi_len = 14
    if len(df_1d) < rsi_len:
        return np.zeros(n)
    
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_len, min_periods=rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len, min_periods=rsi_len).mean()
    rs = gain / loss
    rsi_1d = (100 - (100 / (1 + rs))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
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
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: RSI relative to 50
        bullish = rsi_1d_aligned[i] > 50
        bearish = rsi_1d_aligned[i] < 50
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + bullish 1d RSI + volume
            if (close[i] > dc_upper[i] and 
                bullish and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + bearish 1d RSI + volume
            elif (close[i] < dc_lower[i] and 
                  bearish and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 50 RSI level or breaks below Donchian lower
            if rsi_1d_aligned[i] < 50 or close[i] < dc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 50 RSI level or breaks above Donchian upper
            if rsi_1d_aligned[i] > 50 or close[i] > dc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_RSI14_Donchian_Volume_v1"
timeframe = "4h"
leverage = 1.0