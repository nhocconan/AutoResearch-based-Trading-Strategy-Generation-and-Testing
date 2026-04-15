#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R extremes combined with 12h EMA20 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions on daily timeframe for mean reversion entries.
# EMA20 on 12h provides trend filter to avoid counter-trend trades in strong moves.
# Volume confirmation ensures momentum validity. Designed for low trade frequency (15-25/year) to minimize fee drag.
# Works in both bull and bear markets: mean reversion in ranges, trend-filtered entries in trends.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 1d and 12h HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 20 or len(df_12h) < 20:
        return np.zeros(n)
    
    # === 1d Indicators: Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - df_1d['close'].values) / (highest_high_1d - lowest_low_1d + 1e-10) * -100
    
    # Align to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # === 12h Indicators: EMA(20) for trend filter ===
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
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
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # Williams %R oversold (< -80) AND price above EMA20 (bullish bias)
        if williams_r_aligned[i] < -80 and close[i] > ema_20_12h_aligned[i] and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Williams %R overbought (> -20) AND price below EMA20 (bearish bias)
        elif williams_r_aligned[i] > -20 and close[i] < ema_20_12h_aligned[i] and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_WilliamsR_EMA20_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0