#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian_20_Squeeze_Breakout_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel and squeeze detection
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low for Donchian channel (20-day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channel upper and lower bounds (20-day)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Bollinger Bands for squeeze detection (20, 2)
    close_1d = df_1d['close'].values
    close_series_1d = pd.Series(close_1d)
    bb_mid = close_series_1d.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series_1d.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Squeeze condition: Bollinger Band width below 20-period average (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma
    
    # Align Donchian levels and squeeze condition to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-period average (12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        squeeze = squeeze_aligned[i] > 0.5  # True if squeeze condition
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above upper Donchian band after low volatility squeeze
            if price > upper and squeeze and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian band after low volatility squeeze
            elif price < lower and squeeze and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below lower Donchian band (reverse signal)
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above upper Donchian band (reverse signal)
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals