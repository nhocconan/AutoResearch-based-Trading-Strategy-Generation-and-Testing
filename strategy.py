#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R (14) for overbought/oversold extremes
# combined with 1d EMA(50) trend filter and 6h volume confirmation.
# In bear markets (price < EMA50), short when Williams %R > -20 (overbought bounce).
# In bull markets (price > EMA50), long when Williams %R < -80 (oversold bounce).
# Volume filter ensures momentum validity. Designed for low trade frequency 
# (12-30/year) to minimize fee drag while adapting to trend via EMA50.

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Williams %R (14) and EMA(50) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - df_1d['close'].values) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero (when high == low)
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    
    # 1d EMA(50) for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Bull market: price above EMA50
        # Bear market: price below EMA50
        is_bull = close[i] > ema_50_1d_aligned[i]
        is_bear = close[i] < ema_50_1d_aligned[i]
        
        # === LONG CONDITIONS ===
        # Only in bull market: long when oversold (Williams %R < -80)
        if is_bull and williams_r_aligned[i] < -80 and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Only in bear market: short when overbought (Williams %R > -20)
        elif is_bear and williams_r_aligned[i] > -20 and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR_EMA50_VolFilter_v1"
timeframe = "6h"
leverage = 1.0