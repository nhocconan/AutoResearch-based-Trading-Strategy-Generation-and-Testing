#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA200 trend filter.
# Uses price channel breakouts from the 4h timeframe for momentum, volume to confirm breakout strength,
# and a long-term EMA200 from daily timeframe to filter trend direction. Designed to work in both bull
# and bear markets by following the higher timeframe trend. Target: 20-30 trades per year to avoid fee drag.

name = "4h_Donchian20_Breakout_Volume_EMA200"
timeframe = "4h"
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
    
    # Load 1D data ONCE for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMA200 to 4H timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 20-period average
    volume_s = pd.Series(volume)
    vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to EMA200
        price_above_ema = close[i] > ema200_1d_aligned[i]
        price_below_ema = close[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # LONG: Break above Donchian high with volume and uptrend
            if (close[i] > donchian_high[i]) and price_above_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low with volume and downtrend
            elif (close[i] < donchian_low[i]) and price_below_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below Donchian low or trend changes
            if (close[i] < donchian_low[i]) or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above Donchian high or trend changes
            if (close[i] > donchian_high[i]) or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals