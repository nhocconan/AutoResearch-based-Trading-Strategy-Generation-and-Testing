#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses weekly Camarilla levels (R4/S4) for institutional bias: long only above weekly R4, short only below weekly S4
# Uses 6h volume > 1.8x 20-period EMA for confirmation to avoid false breakouts
# Designed for 6h timeframe targeting 15-30 trades/year with discrete sizing (0.25)
# Weekly pivot filter reduces counter-trend trades, volume confirmation increases breakout validity
# Works in bull markets (breakouts above weekly R4 with volume) and bear markets (breakdowns below weekly S4 with volume)

name = "6h_Donchian20_WeeklyCamarillaR4S4_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 6h data for Donchian and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume EMA(20) for confirmation
    vol_6h = df_6h['volume'].values
    vol_series = pd.Series(vol_6h)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly Camarilla levels: R4, S4 from weekly OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_range = high_1w - low_1w
    r4 = close_1w + 1.1 * camarilla_range
    s4 = close_1w - 1.1 * camarilla_range
    
    # Align indicators to 6h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_6h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_6h, lowest_20)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ema_20)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(vol_ema_20_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.8 x 20-period EMA
        volume_confirmed = volume[i] > (1.8 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: price breaks above Donchian high + above weekly R4 + volume confirmation
            if (close[i] > highest_20_aligned[i] and 
                close[i] > r4_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below weekly S4 + volume confirmation
            elif (close[i] < lowest_20_aligned[i] and 
                  close[i] < s4_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian low (breakdown) OR below weekly S4 (bearish shift)
            if close[i] < lowest_20_aligned[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above Donchian high (breakout) OR above weekly R4 (bullish shift)
            if close[i] > highest_20_aligned[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals