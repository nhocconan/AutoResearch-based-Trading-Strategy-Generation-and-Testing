#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian breakout with volume confirmation.
# Uses Choppiness Index on 1d to detect ranging markets (filter out false breakouts).
# Breakouts from 1d Donchian channel require volume spike and trend alignment.
# Designed to work in both bull and bear markets by filtering out choppy conditions.
# Target: 20-30 trades/year to stay within optimal range.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index, Donchian channels, and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Choppiness Index (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0  # First value has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(np.sum(atr14, axis=0) / (highest_high_14 - lowest_low_14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high_14 - lowest_low_14) == 0, 50, chop)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Calculate 1d Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, high_max_20)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, low_min_20)
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average (strict to reduce trades)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 55  # Need 20-period Donchian (1d) + EMA34 (1d) + volume MA20 + Chop (14-period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_4h[i]) or 
            np.isnan(donchian_high_4h[i]) or 
            np.isnan(donchian_low_4h[i]) or 
            np.isnan(ema34_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Only trade when market is trending (Choppiness Index < 38.2)
        trending_market = chop_4h[i] < 38.2
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema34_4h[i]
        price_below_ema = close[i] < ema34_4h[i]
        
        # Price relative to 1d Donchian channels
        price_above_high = close[i] > donchian_high_4h[i]
        price_below_low = close[i] < donchian_low_4h[i]
        
        if position == 0:
            # Long: Price breaks above 1d Donchian high with volume, trend alignment, and trending market
            if (price_above_high and price_above_ema and volume_filter and trending_market):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1d Donchian low with volume, trend alignment, and trending market
            elif (price_below_low and price_below_ema and volume_filter and trending_market):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 1d Donchian low OR below 1d EMA34 OR market becomes choppy
            if (close[i] < donchian_low_4h[i]) or (close[i] < ema34_4h[i]) or (chop_4h[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above 1d Donchian high OR above 1d EMA34 OR market becomes choppy
            if (close[i] > donchian_high_4h[i]) or (close[i] > ema34_4h[i]) or (chop_4h[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_Donchian20_Volume_EMA34"
timeframe = "4h"
leverage = 1.0