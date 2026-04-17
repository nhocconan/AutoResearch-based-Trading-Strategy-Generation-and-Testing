#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND 12h EMA34 > prior 12h EMA34 AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND 12h EMA34 < prior 12h EMA34 AND volume > 1.5x 20-period average.
Exit when price reverts to Donchian middle band (20-period mean) or volume drops below average.
Uses proven Donchian breakout structure with HTF trend alignment to minimize false breakouts.
Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag.
Works in bull markets (trend continuation) and bear markets (trend reversals via short signals).
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
    
    # Calculate Donchian channels on primary 6h timeframe
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
    
    def rolling_mean(arr, window):
        """Rolling mean"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    upper_band = rolling_max(high, 20)
    lower_band = rolling_min(low, 20)
    middle_band = rolling_mean(close, 20)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA34
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h volume 20-period average
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    middle_aligned = align_htf_to_ltf(prices, df_12h, middle_band)
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        middle = middle_aligned[i]
        ema34 = ema34_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Need prior EMA34 value for trend direction
            if i > start_idx and not np.isnan(ema34_aligned[i-1]):
                prev_ema34 = ema34_aligned[i-1]
                # Long: break above upper band AND EMA34 rising AND volume spike
                if price > upper and ema34 > prev_ema34 and vol > 1.5 * vol_ma:
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower band AND EMA34 falling AND volume spike
                elif price < lower and ema34 < prev_ema34 and vol > 1.5 * vol_ma:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price reverts to middle band OR volume drops below average
            if price <= middle or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reverts to middle band OR volume drops below average
            if price >= middle or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMA34_Volume"
timeframe = "6h"
leverage = 1.0