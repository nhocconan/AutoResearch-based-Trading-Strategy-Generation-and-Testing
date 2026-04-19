#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h Volume and ADX filter.
# Use 12h ADX > 25 to filter for trending markets, volume > 1.5x 20-period average for confirmation.
# Long when price breaks above 4h Donchian(20) high, short when breaks below low.
# Exit when price crosses the Donchian midline.
# Designed to work in both bull and bear markets by filtering for strong trends.
# Target: 20-40 trades/year per symbol to stay within frequency limits.
name = "4h_Donchian_ADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period)
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate directional movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr = np.zeros(len(close_12h))
    tr[0] = high_12h[0] - low_12h[0]  # First TR
    for i in range(1, len(close_12h)):
        tr[i] = true_range(high_12h[i], low_12h[i], close_12h[i-1])
    
    # Wilder's smoothing (14-period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.sum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    # Smooth TR, +DM, -DM
    atr_12h = wilders_smoothing(tr, 14)
    plus_di_12h = wilders_smoothing(plus_dm, 14)
    minus_di_12h = wilders_smoothing(minus_dm, 14)
    
    # Calculate DX and ADX
    dx = np.zeros_like(close_12h)
    dx_sum = plus_di_12h + minus_di_12h
    dx = np.where(dx_sum != 0, 100 * np.abs(plus_di_12h - minus_di_12h) / dx_sum, 0)
    adx_12h = wilders_smoothing(dx, 14)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donch_high = rolling_max(high_4h, 20)
    donch_low = rolling_min(low_4h, 20)
    
    # Align indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Get 4h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure ADX (14*2+6), Donchian (20), and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Trend strength filter
        strong_trend = adx_val > 25
        
        if position == 0:
            # Enter on Donchian breakout with volume and trend confirmation
            if strong_trend and volume_confirmed:
                if price > donch_high_val:
                    signals[i] = 0.25
                    position = 1
                elif price < donch_low_val:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses Donchian midline
            midline = (donch_high_val + donch_low_val) / 2
            if price < midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses Donchian midline
            midline = (donch_high_val + donch_low_val) / 2
            if price > midline:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals