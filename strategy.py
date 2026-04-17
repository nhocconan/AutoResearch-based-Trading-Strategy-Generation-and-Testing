#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation.
Long when price breaks above 20-period Donchian high AND 1d EMA50 is rising AND volume > 1.3x 20-period average.
Short when price breaks below 20-period Donchian low AND 1d EMA50 is falling AND volume > 1.3x 20-period average.
Exit when price touches opposite Donchian band (mean reversion) or volume drops below average.
Designed for low trade frequency (12-37/year) on 12h timeframe with strong trend and volume filters.
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
    
    # Calculate Donchian channels (20-period) on 12h
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high = rolling_max(high_12h, 20)
    donchian_low = rolling_min(low_12h, 20)
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA50 slope (trend direction) on 1d
    ema50_slope = np.zeros_like(ema50_1d)
    ema50_slope[1:] = ema50_1d[1:] - ema50_1d[:-1]
    
    # Calculate volume average (20-period) on 12h
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        ema50 = ema50_aligned[i]
        ema50_slope_val = ema50_slope_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high AND EMA50 rising AND volume > 1.3x avg
            if price > donch_high and ema50_slope_val > 0 and vol > 1.3 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND EMA50 falling AND volume > 1.3x avg
            elif price < donch_low and ema50_slope_val < 0 and vol > 1.3 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price touches Donchian low (mean reversion) OR volume < average
            if price <= donch_low or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price touches Donchian high (mean reversion) OR volume < average
            if price >= donch_high or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0