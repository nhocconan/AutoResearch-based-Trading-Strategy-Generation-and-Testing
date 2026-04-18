#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index + Donchian(20) breakout with volume confirmation.
# In trending markets (CHOP < 38.2), breakouts continue; in ranging markets (CHOP > 61.8), avoid breakouts.
# Volume ensures institutional participation. Designed for low trade frequency to minimize fee drag.
# Target: 20-40 trades/year per symbol.
name = "4h_Chop_Donchian20_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate True Range for Choppiness Index
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ADX-like component: sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    # Calculate sum of absolute price changes over 14 periods
    close_diff = np.abs(np.diff(df_1d['close'].values, prepend=np.nan))
    sum_close_diff = pd.Series(close_diff).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(sum(tr) / (max(high) - min(low))) / log10(14)
    # Avoid division by zero
    price_range = highest_high_14 - lowest_low_14
    chop_raw = 100 * np.log10(atr_sum / price_range) / np.log10(14)
    # Handle cases where price_range is zero or very small
    chop = np.where(price_range > 0, chop_raw, 50.0)  # Default to neutral when no range
    chop = np.nan_to_num(chop, nan=50.0)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 20-period Donchian channels on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Choppiness regime filter: only trade when trending (CHOP < 38.2)
        trend_regime = chop_aligned[i] < 38.2
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        trade_allowed = trend_regime and vol_confirm
        
        if position == 0:
            # Long: price breaks above 20-period high
            if trade_allowed and close[i] > highest_high[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low
            elif trade_allowed and close[i] < lowest_low[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 10-period low (faster exit)
            lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
            if not np.isnan(lowest_low_10[i]) and close[i] < lowest_low_10[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 10-period high
            highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
            if not np.isnan(highest_high_10[i]) and close[i] > highest_high_10[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals