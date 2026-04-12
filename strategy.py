#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on daily data
    # Upper channel: highest high of last 20 days
    high_series = pd.Series(high_1d)
    donchian_up = high_series.rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 days
    low_series = pd.Series(low_1d)
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donchian_up_aligned = align_htf_to_ltf(prices, df_1d, donchian_up)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate Choppiness Index on daily data (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR (smoothed TR)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = high_series.rolling(window=14, min_periods=14).max().values
    ll = low_series.rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    
    # Align Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter - 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(donchian_up_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Chop > 61.8 indicates ranging market (mean reversion opportunity)
        # Chop < 38.2 indicates trending market
        # We'll use Chop > 61.8 for mean reversion at Donchian levels
        chop_high = chop_aligned[i] > 61.8  # Ranging market
        
        # Long signal: price touches or breaks below lower Donchian in ranging market
        long_signal = chop_high and (low[i] <= donchian_low_aligned[i]) and volume_ok[i]
        # Short signal: price touches or breaks above upper Donchian in ranging market
        short_signal = chop_high and (high[i] >= donchian_up_aligned[i]) and volume_ok[i]
        
        # Exit when price moves back to middle of channel or Chop drops
        mid_channel = (donchian_up_aligned[i] + donchian_low_aligned[i]) / 2
        exit_long = chop_aligned[i] < 50 or close[i] >= mid_channel
        exit_short = chop_aligned[i] < 50 or close[i] <= mid_channel
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals