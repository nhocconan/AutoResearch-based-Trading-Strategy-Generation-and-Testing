#!/usr/bin/env python3
# 6h_12h_donchian_volume_regime_v1
# Hypothesis: Combine 12h Donchian breakout (20-period) with volume confirmation and 12h ADX regime filter on 6h timeframe.
# In trending markets (ADX > 25), we take breakouts in the direction of the trend.
# In ranging markets (ADX <= 25), we fade at Donchian bands with volume confirmation.
# This adapts to both bull and bear markets by using trend-following in trends and mean-reversion in ranges.
# Volume filter ensures we only act on significant moves.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_12h_donchian_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Donchian and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # 12h ADX for regime detection (14-period)
    # Calculate True Range components
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - df_12h['close'].values[:-1])
    tr3 = np.abs(low_12h[1:] - df_12h['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(20, 20) + 14 + 5  # Donchian + ADX + buffer
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below midpoint or trend fails
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] < midpoint or (adx_aligned[i] > 25 and close[i] < donchian_low_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above midpoint or trend fails
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] > midpoint or (adx_aligned[i] > 25 and close[i] > donchian_high_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if adx_aligned[i] > 25:  # Trending regime
                # Breakout entries in direction of Donchian breakout
                if close[i] > donchian_high_aligned[i] and volume_filter:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < donchian_low_aligned[i] and volume_filter:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging regime
                # Fade at Donchian bands with volume confirmation
                if close[i] <= donchian_low_aligned[i] and volume_filter:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= donchian_high_aligned[i] and volume_filter:
                    position = -1
                    signals[i] = -0.25
    
    return signals