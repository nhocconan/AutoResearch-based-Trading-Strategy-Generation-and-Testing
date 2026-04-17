#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian breakout with 1w EMA trend filter and volume confirmation.
Long when price breaks above Donchian(20) upper band AND 1w EMA50 is rising AND volume > 1.5x 20-period average.
Short when price breaks below Donchian(20) lower band AND 1w EMA50 is falling AND volume > 1.5x 20-period average.
Exit when price crosses Donchian middle band (mean of upper/lower) or volume drops below average.
Uses proven Donchian breakout structure with multi-timeframe trend filter to work in both bull and bear markets.
Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Get 1w data for EMA trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 12h
    def rolling_max(arr, window):
        """Rolling maximum"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        """Rolling minimum"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_12h = rolling_max(high_12h, 20)
    lower_12h = rolling_min(low_12h, 20)
    middle_12h = (upper_12h + lower_12h) / 2.0
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_rising = np.diff(ema50_1w, prepend=ema50_1w[0]) > 0  # True if rising
    ema50_1w_falling = np.diff(ema50_1w, prepend=ema50_1w[0]) < 0  # True if falling
    
    # Calculate volume average (20-period) on 12h
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    middle_aligned = align_htf_to_ltf(prices, df_12h, middle_12h)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema50_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_rising.astype(float))
    ema50_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_falling.astype(float))
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_1w_rising_aligned[i]) or np.isnan(ema50_1w_falling_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        middle = middle_aligned[i]
        ema_trend = ema50_1w_aligned[i]
        ema_rising = bool(ema50_1w_rising_aligned[i])
        ema_falling = bool(ema50_1w_falling_aligned[i])
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Price > upper band AND 1w EMA50 rising AND volume > 1.5x avg
            if price > upper and ema_rising and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Price < lower band AND 1w EMA50 falling AND volume > 1.5x avg
            elif price < lower and ema_falling and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price < middle band OR volume < average
            if price < middle or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price > middle band OR volume < average
            if price > middle or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DonchianBreakout_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0