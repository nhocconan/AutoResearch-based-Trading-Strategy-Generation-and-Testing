#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h HMA(21) for trend direction and 1d Williams %R(14) for mean reversion timing.
# In 12h uptrend (price > HMA), wait for 1d Williams %R < -80 (oversold) to go long.
# In 12h downtrend (price < HMA), wait for 1d Williams %R > -20 (overbought) to go short.
# Volume confirmation ensures momentum validity. Designed for low trade frequency (20-40/year) to minimize fee drag.
# Williams %R is a momentum oscillator that works well in ranging markets, complementing HMA trend filter.

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
    
    # Get 12h and 1d HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: HMA(21) ===
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    n = 21
    half_n = n // 2
    sqrt_n = int(np.sqrt(n))
    
    wma_half = np.array([np.nan] * len(close_12h))
    wma_full = np.array([np.nan] * len(close_12h))
    wma_sqrt = np.array([np.nan] * len(close_12h))
    
    if len(close_12h) >= half_n:
        wma_half[half_n-1:] = wma(close_12h, half_n)
    if len(close_12h) >= n:
        wma_full[n-1:] = wma(close_12h, n)
    if len(close_12h) >= sqrt_n:
        wma_sqrt[sqrt_n-1:] = wma(close_12h, sqrt_n)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    # WMA(sqrt(n)) of diff
    hma_12h = np.array([np.nan] * len(close_12h))
    if len(diff) >= sqrt_n:
        hma_12h[sqrt_n-1:] = wma(diff, sqrt_n)
    
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 1d Indicators: Williams %R(14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
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
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. In 12h uptrend (price > 12h HMA)
        # 2. 1d Williams %R < -80 (oversold)
        # 3. Volume confirmation
        if (close[i] > hma_12h_aligned[i]) and (williams_r_aligned[i] < -80) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. In 12h downtrend (price < 12h HMA)
        # 2. 1d Williams %R > -20 (overbought)
        # 3. Volume confirmation
        elif (close[i] < hma_12h_aligned[i]) and (williams_r_aligned[i] > -20) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_HMA21_WilliamsR14_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0