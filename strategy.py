#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and 1d ADX trend filter.
# Long when price breaks above 12h Donchian high, 1d ADX > 25 (trending), and volume > 1.5x average.
# Short when price breaks below 12h Donchian low, 1d ADX > 25, and volume confirmation.
# Exit when price returns to 12h Donchian midpoint or ADX < 20 (range).
# Designed to capture trends in both bull and bear markets with low trade frequency to minimize fee drag.
# Target: 20-25 trades/year per symbol (80-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels (20-period) on 12h
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Load 1d data ONCE for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[np.nan], low_1d[:-1] - low_1d[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_period = 14
    atr_1d = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    plus_di_1d = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values / atr_1d * 100
    minus_di_1d = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values / atr_1d * 100
    dx = np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d) * 100
    dx = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx)
    adx_1d = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align indicators to lower timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 30)  # Need Donchian and ADX
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 for trending market
        trending = adx_1d_aligned[i] > 25
        ranging = adx_1d_aligned[i] < 20  # Exit condition
        
        if position == 0:
            # Look for Donchian breakouts
            # Long: price breaks above Donchian high AND trending
            if (close[i] > donchian_high_aligned[i] and 
                trending and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND trending
            elif (close[i] < donchian_low_aligned[i] and 
                  trending and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midpoint or market becomes ranging
            if (close[i] <= donchian_mid_aligned[i] or 
                ranging):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midpoint or market becomes ranging
            if (close[i] >= donchian_mid_aligned[i] or 
                ranging):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12hDonchian_1dADX_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0