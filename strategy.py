#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets, while EMA200 filter ensures
# alignment with long-term trend to avoid counter-trend trades. Volume confirmation
# filters false breakouts. Designed to work in both bull (breakouts above upper band) 
# and bear (breakdowns below lower band) markets with proper trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Donchian(20) channels ===
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 4h volume confirmation ===
    vol_avg20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    warmup = 200  # EMA200 needs 200 periods
    position = 0
    
    for i in range(warmup, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(vol_avg20[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        vol_filter = volume[i] > 1.5 * vol_avg20[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band + 1d uptrend + volume filter
            if close[i] > high_max_20[i] and close[i] > ema200_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian band + 1d downtrend + volume filter
            if close[i] < low_min_20[i] and close[i] < ema200_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long on breakdown below lower band or trend reversal
            if close[i] < low_min_20[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on breakout above upper band or trend reversal
            if close[i] > high_max_20[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA200_VolumeFilter"
timeframe = "4h"
leverage = 1.0