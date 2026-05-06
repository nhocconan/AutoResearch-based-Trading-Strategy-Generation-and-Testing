#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation
# Uses weekly trend filter (price > 50-week SMA) to avoid counter-trend trades
# Choppiness Index > 61.8 = ranging (mean revert at Donchian bands), < 38.2 = trending (breakout follow)
# Volume confirmation (>1.5x 20-bar average) ensures breakout strength
# Designed for low frequency: target 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear: trend filter avoids whipsaw, regime filter adapts to market conditions

name = "12h_Chop_Donchian20_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly trend filter: price > 50-week SMA
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Daily Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop calculation: 100 * log(sum(ATR14)/ (max(high)-min(low)) ) / log(14)
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log(sum_atr / (max_high - min_low)) / np.log(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(sma50_1w_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime: Chop > 61.8 = ranging, Chop < 38.2 = trending
            is_ranging = chop_aligned[i] > 61.8
            is_trending = chop_aligned[i] < 38.2
            
            if is_ranging:
                # Mean reversion at Donchian bands in ranging markets
                if close[i] <= donchian_low[i] and close[i] > sma50_1w_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= donchian_high[i] and close[i] < sma50_1w_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            elif is_trending:
                # Breakout follow in trending markets with weekly trend filter
                if close[i] > donchian_high[i] and close[i] > sma50_1w_aligned[i] and volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i] and close[i] < sma50_1w_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses Donchian midpoint or trend fails
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] <= midpoint or close[i] < sma50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian midpoint or trend fails
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] >= midpoint or close[i] > sma50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals