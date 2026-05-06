#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Choppiness Index regime filter with 
# 12h Donchian(20) breakout and volume confirmation.
# Long when price breaks above Donchian(20) upper band in trending market (CHOP < 38.2) 
# with volume > 1.5x average.
# Short when price breaks below Donchian(20) lower band in trending market (CHOP < 38.2) 
# with volume > 1.5x average.
# Exit when price crosses Donchian midpoint or when market becomes choppy (CHOP > 61.8).
# Uses daily timeframe for regime detection (more robust than intraday) and 
# 12h for execution to minimize trade frequency and fee drag.
# Target: 12-37 trades per year (50-150 over 4 years) with 0.25 position sizing.

name = "12h_1dChop_Trend_Donchian20_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) on 12h high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate daily Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over 14 periods
    atr_sum = tr.rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = df_1d['high'].rolling(window=14, min_periods=14).max().values
    ll = df_1d['low'].rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: >1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * vol_ma_30)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filters
        is_trending = chop_aligned[i] < 38.2   # Trending market
        is_choppy = chop_aligned[i] > 61.8     # Choppy/ranging market
        
        if position == 0:
            # Enter long: Donchian breakout above upper band in trending market with volume
            if close[i] > donchian_high[i] and is_trending and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Donchian breakdown below lower band in trending market with volume
            elif close[i] < donchian_low[i] and is_trending and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses Donchian midpoint OR market becomes choppy
            if close[i] < donchian_mid[i] or is_choppy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian midpoint OR market becomes choppy
            if close[i] > donchian_mid[i] or is_choppy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals