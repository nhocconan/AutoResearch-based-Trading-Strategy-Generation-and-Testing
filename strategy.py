#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian breakout with 1w trend filter and volume confirmation.
Long when price breaks above Donchian(20) high AND 1w EMA50 is rising AND volume > 1.5x 20-period average.
Short when price breaks below Donchian(20) low AND 1w EMA50 is falling AND volume > 1.5x 20-period average.
Exit when price returns to Donchian midpoint or volume drops below average.
Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag.
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
    
    # Get 12h data for Donchian calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 12h
    def rolling_max(arr, window):
        """Rolling maximum"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        """Rolling minimum"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high_12h, 20)
    donchian_low = rolling_min(low_12h, 20)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_prev = np.roll(ema50_1w, 1)
    ema50_1w_prev[0] = np.nan
    ema50_rising = ema50_1w > ema50_1w_prev
    ema50_falling = ema50_1w < ema50_1w_prev
    
    # Calculate volume average (20-period) on 12h
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema50_falling)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        d_high = donchian_high_aligned[i]
        d_low = donchian_low_aligned[i]
        d_mid = donchian_mid_aligned[i]
        ema50_up = ema50_rising_aligned[i]
        ema50_down = ema50_falling_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high AND 1w EMA50 rising AND volume > 1.5x avg
            if price > d_high and ema50_up and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND 1w EMA50 falling AND volume > 1.5x avg
            elif price < d_low and ema50_down and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to Donchian midpoint OR volume < average
            if price <= d_mid or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to Donchian midpoint OR volume < average
            if price >= d_mid or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0