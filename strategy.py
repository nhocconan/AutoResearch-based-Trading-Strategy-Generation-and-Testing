# 12h_Donchian_Breakout_1dTrend_Volume_v2
# Hypothesis: Donchian breakouts on 12h capture sustained trends in both bull and bear markets, with 1d trend filter avoiding counter-trend trades and volume confirmation reducing false breakouts. This combination yields high-probability trades with low frequency (<30/year) suitable for 12h timeframe, minimizing fee drag while capturing major moves.

name = "12h_Donchian_Breakout_1dTrend_Volume_v2"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period EMA for 1D trend
    close_1d_series = pd.Series(close_1d)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-period average volume for 1D
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1D indicators to 12H timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Calculate 20-period Donchian channels on 12H
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1D EMA
        uptrend = close[i] > ema20_1d_aligned[i]
        downtrend = close[i] < ema20_1d_aligned[i]
        
        # Volume confirmation: current volume > 20-period average
        volume_confirm = volume[i] > vol_ma20_1d_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + uptrend + volume confirmation
            if close[i] > high_max_20[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + downtrend + volume confirmation
            elif close[i] < low_min_20[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian or trend reverses
            if close[i] < low_min_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian or trend reverses
            if close[i] > high_max_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals