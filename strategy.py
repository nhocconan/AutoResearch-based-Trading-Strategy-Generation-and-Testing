#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h EMA crossover (8/21) for trend, combined with 1d Donchian(20) breakout
# and volume confirmation. In trending markets (price above/below Donchian bands), trade in EMA trend direction.
# Volume filter ensures momentum validity. Designed for low trade frequency (15-35/year) to minimize fee drag
# while adapting to regime via Donchian structure and EMA trend filter. Works in bull/bear via trend filter.

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
    
    # === 12h Indicators: EMA Crossover (8/21) ===
    ema_8_12h = pd.Series(df_12h['close'].values).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_8_aligned = align_htf_to_ltf(prices, df_12h, ema_8_12h)
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # === 1d Indicators: Donchian Channel (20) ===
    high_20_1d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20_1d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20_1d)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_8_aligned[i]) or np.isnan(ema_21_aligned[i]) or
            np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Trending market: price outside Donchian bands
        # Ranging market: price inside Donchian bands (no trade)
        
        above_upper = close[i] > high_20_aligned[i]
        below_lower = close[i] < low_20_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. Price above upper Donchian band (uptrend)
        # 2. EMA 8 > EMA 21 (bullish momentum)
        # 3. Volume confirmation
        if vol_confirm and above_upper and (ema_8_aligned[i] > ema_21_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price below lower Donchian band (downtrend)
        # 2. EMA 8 < EMA 21 (bearish momentum)
        # 3. Volume confirmation
        elif vol_confirm and below_lower and (ema_8_aligned[i] < ema_21_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat (ranging or no confirmation)
    
    return signals

name = "6h_EMA8_21_Donchian20_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0