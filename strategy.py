#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1d ADX for trend strength, 12h ATR-based Donchian breakout for entry,
# and volume confirmation. Uses 1d ADX > 25 to filter trending markets and avoid whipsaws in ranging conditions.
# Donchian channels provide volatility-adjusted breakout levels. Volume > 1.5x 20-period average confirms breakout strength.
# Exits when price returns to the midpoint of the Donchian channel or trend weakens (ADX < 20).
# Designed to work in both bull and bear markets by using 1d trend filter to avoid counter-trend trades.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    tr_series = pd.Series(tr)
    atr = tr_series.ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx_series = pd.Series(dx)
    adx = dx_series.ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Load 12h data ONCE for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donch_period = 20
    upper_channel = pd.Series(high_12h).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_channel = pd.Series(low_12h).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Align indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    upper_channel_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need Donchian and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(upper_channel_aligned[i]) or
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Look for Donchian breakouts
            # Only trade in trending markets
            
            # Long: price breaks above upper Donchian channel AND trending market
            if (close[i] > upper_channel_aligned[i] and 
                trending and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian channel AND trending market
            elif (close[i] < lower_channel_aligned[i] and 
                  trending and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel or trend weakens
            mid_channel = (upper_channel_aligned[i] + lower_channel_aligned[i]) / 2
            if (close[i] <= mid_channel or 
                adx_aligned[i] < 20):  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian channel or trend weakens
            mid_channel = (upper_channel_aligned[i] + lower_channel_aligned[i]) / 2
            if (close[i] >= mid_channel or 
                adx_aligned[i] < 20):  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dADX_12hDonchian_Breakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0