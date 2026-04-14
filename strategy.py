#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w CCI for trend filter and 1d Donchian breakout for entry.
# Weekly CCI filters trades to align with higher timeframe trend, avoiding counter-trend trades.
# Daily Donchian breakout (20) provides entry with volume confirmation (>1.5x 20-day average volume).
# Designed to work in both bull and bear markets by using 1w trend filter to avoid counter-trend trades.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for CCI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate CCI (20) on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price
    tp = (high_1w + low_1w + close_1w) / 3
    # SMA of typical price
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    # Mean deviation
    md = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # CCI
    cci = (tp - sma_tp) / (0.015 * md)
    # Handle division by zero or near-zero md
    cci = np.where(md == 0, 0, cci)
    
    # Align 1w CCI to 1d timeframe
    cci_1w_aligned = align_htf_to_ltf(prices, df_1w, cci)
    
    # Load 1d data ONCE for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian channels to 1d timeframe (no change, but for consistency)
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
        if (np.isnan(cci_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts above 1d Donchian high or below 1d Donchian low
            # Only trade in direction of 1w CCI (>100 for uptrend, <-100 for downtrend)
            
            # Long: price breaks above 1d Donchian high AND 1w CCI > 100 (uptrend)
            if (close[i] > donchian_high_aligned[i] and 
                cci_1w_aligned[i] > 100 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 1d Donchian low AND 1w CCI < -100 (downtrend)
            elif (close[i] < donchian_low_aligned[i] and 
                  cci_1w_aligned[i] < -100 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 1d Donchian low or 1w CCI turns negative
            if (close[i] <= donchian_low_aligned[i] or 
                cci_1w_aligned[i] < 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 1d Donchian high or 1w CCI turns positive
            if (close[i] >= donchian_high_aligned[i] or 
                cci_1w_aligned[i] > 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wCCI_1dDonchian_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0