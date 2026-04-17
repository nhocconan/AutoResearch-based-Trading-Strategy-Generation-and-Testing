#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1d Donchian(20) breakout with volume confirmation.
# Choppiness Index (14) > 61.8 indicates ranging market (mean reversion), < 38.2 indicates trending.
# In trending regime (CHOP < 38.2): trade breakouts of 1d Donchian channels.
# In ranging regime (CHOP > 61.8): fade extremes at 1d Donchian channels.
# Volume spike confirms breakout/breakdown strength.
# Designed to work in both bull (trend-following) and bear/range (mean-reversion) markets.
# Target: 15-35 trades/year to stay within optimal range for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period True Range for Choppiness Index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Calculate 14-period ATR (smoothed TR)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Calculate Choppiness Index: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_raw = 100 * np.log10(sum_tr14 / (hh14 - ll14)) / np.log10(14)
    # Handle division by zero when hh14 == ll14
    chop_raw = np.where((hh14 - ll14) == 0, 100, chop_raw)
    
    # Calculate 1d Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 12h
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_raw)
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, high_max_20)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, low_min_20)
    
    # Volume filter: current volume > 1.5 * 20-period average (moderate to balance signal quality)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 20-period Donchian + 14-period Chop + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_12h[i]) or 
            np.isnan(donchian_high_12h[i]) or 
            np.isnan(donchian_low_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        chop = chop_12h[i]
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Price relative to 1d Donchian channels
        price_above_high = close[i] > donchian_high_12h[i]
        price_below_low = close[i] < donchian_low_12h[i]
        
        if position == 0:
            # Trending market (CHOP < 38.2): breakout strategy
            if chop < 38.2:
                # Long: Price breaks above 1d Donchian high with volume
                if price_above_high and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below 1d Donchian low with volume
                elif price_below_low and volume_filter:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (CHOP > 61.8): mean reversion strategy
            elif chop > 61.8:
                # Long: Price pulls back to 1d Donchian low with volume
                if price_below_low and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: Price pulls back to 1d Donchian high with volume
                elif price_above_high and volume_filter:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions
            if chop < 38.2:  # Trending: exit on breakdown
                if close[i] < donchian_low_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging: exit at midpoint or opposite extreme
                mid = (donchian_high_12h[i] + donchian_low_12h[i]) / 2
                if close[i] >= mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            if chop < 38.2:  # Trending: exit on breakout
                if close[i] > donchian_high_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging: exit at midpoint or opposite extreme
                mid = (donchian_high_12h[i] + donchian_low_12h[i]) / 2
                if close[i] <= mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Chop_Donchian20_Volume_Regime"
timeframe = "12h"
leverage = 1.0