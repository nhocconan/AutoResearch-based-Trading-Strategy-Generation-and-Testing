#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation.
Long when price breaks above 20-day high AND price > 1w EMA50 AND volume > 1.5x 20-day average.
Short when price breaks below 20-day low AND price < 1w EMA50 AND volume > 1.5x 20-day average.
Exit when price reverts to 10-day MA or volume drops below average.
Designed for low trade frequency (7-25/year) on 1d timeframe to minimize fee drag and capture medium-term trends.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicator calculation (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 1d
    def rolling_max(arr, window):
        """Rolling maximum"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        """Rolling minimum"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high_1d, 20)
    donchian_low = rolling_min(low_1d, 20)
    
    # Calculate 10-day MA for exit
    ma_10 = pd.Series(close_1d).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 20-day volume average
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ma_10_aligned = align_htf_to_ltf(prices, df_1d, ma_10)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ma_10_aligned[i]) or np.isnan(volume_ma_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ma10 = ma_10_aligned[i]
        vol_ma = volume_ma_20_aligned[i]
        ema50 = ema_50_1w_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: break above upper Donchian AND price > 1w EMA50 AND volume > 1.5x avg
            if price > upper and price > ema50 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian AND price < 1w EMA50 AND volume > 1.5x avg
            elif price < lower and price < ema50 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 10-day MA OR volume drops below average
            if price < ma10 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 10-day MA OR volume drops below average
            if price > ma10 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0