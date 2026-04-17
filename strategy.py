#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h Donchian(20) breakout + volume confirmation + ATR volatility filter.
Long when price breaks above 4h Donchian high with volume > 1.3x 20-period average and current ATR < 1.5x 20-period ATR average (low volatility breakout).
Short when price breaks below 4h Donchian low with volume > 1.3x 20-period average and current ATR < 1.5x 20-period ATR average.
Exit when price returns to the 4h Donchian midpoint.
Uses 4h/1d for signal direction, 1h only for entry timing. Session filter (08-20 UTC) to reduce noise.
Position size 0.20 to limit fee drag. Target: 60-150 total trades over 4 years.
Works in bull markets (trend continuation) and bear markets (mean reversion after low volatility periods).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels, volume, and ATR
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    upper_20 = high_series.rolling(window=20, min_periods=20).max().values
    lower_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR (14-period) for volatility filter
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h volume 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1h
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    atr_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_ma_20)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ATR and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(atr_ma_20_4h_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(volume_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_confirmed = volume_4h_aligned[i] > 1.3 * vol_ma_20_4h_aligned[i]
        
        # Volatility filter: current ATR < 1.5x 20-period ATR average (breakout from low volatility)
        vol_filter = atr_aligned[i] < 1.5 * atr_ma_20_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian high with volume and low volatility
            if (close[i] > upper_20_aligned[i] and 
                volume_confirmed and 
                vol_filter):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low with volume and low volatility
            elif (close[i] < lower_20_aligned[i] and 
                  volume_confirmed and 
                  vol_filter):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 4h Donchian midpoint
            midpoint_20 = (upper_20 + lower_20) / 2
            midpoint_20_aligned = align_htf_to_ltf(prices, df_4h, midpoint_20)
            if close[i] < midpoint_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above 4h Donchian midpoint
            midpoint_20 = (upper_20 + lower_20) / 2
            midpoint_20_aligned = align_htf_to_ltf(prices, df_4h, midpoint_20)
            if close[i] > midpoint_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian20_Volume_VolatilityFilter_Session"
timeframe = "1h"
leverage = 1.0