#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-week pivot points for S/R and 1-day Donchian breakout for trend confirmation.
# Weekly pivot levels act as strong institutional S/R zones. Price breaking above weekly R1 with
# price above daily Donchian upper channel (20) indicates bullish breakout with trend alignment.
# Price breaking below weekly S1 with price below daily Donchian lower channel indicates bearish breakdown.
# Volume confirmation (>1.5x 20-period average) filters false breakouts.
# Designed for low trade frequency (<400 total 4h trades) to minimize fee drag.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by using
# weekly structure and daily trend filter to avoid counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot levels for each week (using previous week's data)
    pivot = np.full(len(df_1w), np.nan)
    r1 = np.full(len(df_1w), np.nan)
    s1 = np.full(len(df_1w), np.nan)
    
    for i in range(1, len(df_1w)):
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        p = (ph + pl + pc) / 3.0
        r1_val = 2 * p - pl
        s1_val = 2 * p - ph
        
        pivot[i] = p
        r1[i] = r1_val
        s1[i] = s1_val
    
    # Align weekly pivot levels to 4h timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Load daily data ONCE for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need Donchian and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts above weekly R1 or below weekly S1
            # Only trade in direction of daily Donchian (trend filter)
            
            # Long: price breaks above weekly R1 AND price above daily Donchian high (bullish)
            if (close[i] > r1_aligned[i] and 
                close[i] > donchian_high_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly S1 AND price below daily Donchian low (bearish)
            elif (close[i] < s1_aligned[i] and 
                  close[i] < donchian_low_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly pivot or breaks below daily Donchian low
            if (close[i] <= pivot_aligned[i] or 
                close[i] < donchian_low_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly pivot or breaks above daily Donchian high
            if (close[i] >= pivot_aligned[i] or 
                close[i] > donchian_high_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1wPivot_1dDonchian_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0