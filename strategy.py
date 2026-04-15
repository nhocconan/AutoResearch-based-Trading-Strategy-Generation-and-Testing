#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R mean reversion with 1w EMA200 trend filter and volume confirmation.
# Williams %R(14) identifies overbought/oversold conditions on 1d timeframe.
# In bull markets (price > 1w EMA200), we take longs when %R crosses above -80 from below.
# In bear markets (price < 1w EMA200), we take shorts when %R crosses below -20 from above.
# Volume confirmation ensures institutional participation. Designed for low trade frequency (12-25/year)
# to minimize fee drag. Works in both bull/bear by adapting to 1w EMA200 trend.

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
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - df_1d['close'].values) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero when high == low
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # === 1w Indicators: EMA(200) for trend filter ===
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
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
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 from below (oversold bounce)
        # 2. 1w price above EMA200 (bullish trend bias)
        # 3. Volume confirmation
        if (williams_r_1d_aligned[i] > -80 and
            williams_r_1d_aligned[i-1] <= -80 and
            close[i] > ema_200_1w_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 from above (overbought rejection)
        # 2. 1w price below EMA200 (bearish trend bias)
        # 3. Volume confirmation
        elif (williams_r_1d_aligned[i] < -20 and
              williams_r_1d_aligned[i-1] >= -20 and
              close[i] < ema_200_1w_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR_MeanRev_EMA200_VolFilter_v1"
timeframe = "6h"
leverage = 1.0