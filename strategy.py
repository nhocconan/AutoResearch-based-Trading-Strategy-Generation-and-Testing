#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour Choppiness regime filter.
# In low-chop (trending) markets, we follow breakouts in direction of 12h trend.
# In high-chop (ranging) markets, we fade mean-reversion at Donchian bands.
# Uses volume confirmation to avoid false breakouts. Designed for both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend and chop
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(34) for trend filter
    ema_len = 34
    if len(df_12h) < ema_len:
        return np.zeros(n)
    ema_12h = pd.Series(df_12h['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 12h Choppiness Index (14 periods)
    chop_len = 14
    if len(df_12h) < chop_len:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    atr_12h = np.zeros(len(close_12h))
    for i in range(1, len(close_12h)):
        tr = max(high_12h[i] - low_12h[i],
                 abs(high_12h[i] - close_12h[i-1]),
                 abs(low_12h[i] - close_12h[i-1]))
        if i == 1:
            atr_12h[i] = tr
        else:
            atr_12h[i] = 0.93 * atr_12h[i-1] + 0.07 * tr  # Wilder smoothing
    sum_atr = pd.Series(atr_12h).rolling(window=chop_len, min_periods=chop_len).sum().values
    max_high = pd.Series(high_12h).rolling(window=chop_len, min_periods=chop_len).max().values
    min_low = pd.Series(low_12h).rolling(window=chop_len, min_periods=chop_len).min().values
    chop_12h = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(chop_len)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Donchian channel (20 periods) on 4h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(60, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(chop_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime: < 38.2 = trending, > 61.8 = ranging
        chop = chop_12h_aligned[i]
        trending = chop < 38.2
        ranging = chop > 61.8
        
        # Trend filter: price relative to 12h EMA34
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long in trending market: breakout above + above EMA + volume
            if (trending and 
                close[i] > dc_upper[i] and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short in trending market: breakdown below + below EMA + volume
            elif (trending and 
                  close[i] < dc_lower[i] and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            # Enter long in ranging market: mean reversion at lower band
            elif (ranging and 
                  close[i] < dc_lower[i] and 
                  volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short in ranging market: mean reversion at upper band
            elif (ranging and 
                  close[i] > dc_upper[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: opposite signal or chop extreme
            if (ranging and close[i] > dc_upper[i]) or \
               (trending and close[i] < ema_12h_aligned[i]) or \
               chop > 80:  # extreme chop
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: opposite signal or chop extreme
            if (ranging and close[i] < dc_lower[i]) or \
               (trending and close[i] > ema_12h_aligned[i]) or \
               chop > 80:  # extreme chop
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Donchian_Chop_Volume_v1"
timeframe = "4h"
leverage = 1.0