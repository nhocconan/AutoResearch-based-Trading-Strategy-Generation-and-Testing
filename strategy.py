#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA200 trend filter and Donchian(20) breakout with volume confirmation.
# In bull markets (price > 1w EMA200), long Donchian breakouts; in bear markets (price < 1w EMA200), short Donchian breakdowns.
# Volume filter ensures momentum validity. Designed for low trade frequency (7-25/year) to minimize fee drag.
# Works in both bull and bear by adapting direction based on 1w trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicators: EMA(200) for trend filter ===
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    # Calculate rolling high/low for 20 periods
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In bull market (price > 1w EMA200)
        # 2. Price breaks above Donchian(20) upper channel
        # 3. Volume confirmation
        if (close[i] > ema_200_1w_aligned[i]) and \
           (close[i] > high_20[i]) and \
           vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In bear market (price < 1w EMA200)
        # 2. Price breaks below Donchian(20) lower channel
        # 3. Volume confirmation
        elif (close[i] < ema_200_1w_aligned[i]) and \
             (close[i] < low_20[i]) and \
             vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_1wEMA200_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0