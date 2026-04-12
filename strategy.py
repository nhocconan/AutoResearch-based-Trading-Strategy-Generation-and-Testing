#!/usr/bin/env python3
"""
1d_1w_Market_Regime_Adaptive
Hypothesis: On daily timeframe, adapt strategy based on weekly market regime (trending vs ranging) to work in both bull and bear markets.
- In trending regimes (ADX > 25): Use Donchian breakout with volume confirmation
- In ranging regimes (ADX <= 25): Use mean reversion at Bollinger Bands with volume filter
- Uses weekly ADX for regime detection to avoid whipsaws
- Designed for low trade frequency (10-25/year) by requiring regime alignment and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Market_Regime_Adaptive"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY ADX FOR REGIME DETECTION ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
            else:
                result[i] = np.nan
        return result
    
    period = 14
    tr14 = wilders_smoothing(tr, period)
    plus_dm14 = wilders_smoothing(plus_dm, period)
    minus_dm14 = wilders_smoothing(minus_dm, period)
    
    # Calculate DI and DX
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    
    # Calculate ADX (smoothed DX)
    adx = wilders_smoothing(dx, period)
    
    # === DAILY INDICATORS ===
    # Donchian Channel (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.nanmax(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.nanmin(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Bollinger Bands (20, 2)
    sma_20 = np.zeros(n)
    sum_ = 0.0
    count = 0
    for i in range(n):
        sum_ += close[i]
        count += 1
        if i >= 20:
            sum_ -= close[i-20]
            count -= 1
        if count > 0:
            sma_20[i] = sum_ / count
        else:
            sma_20[i] = np.nan
    
    # Standard deviation
    std_20 = np.zeros(n)
    sum_sq = 0.0
    for i in range(n):
        sum_sq += (close[i] - sma_20[i]) ** 2 if not np.isnan(sma_20[i]) else 0
        if i >= 20:
            if not np.isnan(sma_20[i-20]):
                sum_sq -= (close[i-20] - sma_20[i-20]) ** 2
        if count > 0:
            std_20[i] = np.sqrt(sum_sq / count) if count > 0 else 0
        else:
            std_20[i] = 0
    
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Volume average (20-period)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    # Align weekly ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(sma_20[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Regime detection: trending if ADX > 25, ranging if ADX <= 25
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] <= 25
        
        if is_trending:
            # TRENDING REGIME: Donchian breakout
            long_breakout = close[i] > donchian_high[i-1] if i > 0 and not np.isnan(donchian_high[i-1]) else False
            short_breakout = close[i] < donchian_low[i-1] if i > 0 and not np.isnan(donchian_low[i-1]) else False
            
            if long_breakout and vol_confirm and position != 1:
                position = 1
                signals[i] = 0.25
            elif short_breakout and vol_confirm and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit when price touches opposite Donchian band
            elif position == 1 and close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
                
        else:  # ranging regime
            # RANGING REGIME: Mean reversion at Bollinger Bands
            long_setup = close[i] < bb_lower[i] and vol_confirm
            short_setup = close[i] > bb_upper[i] and vol_confirm
            
            if long_setup and position != -1:  # Allow flipping from short to long
                position = 1
                signals[i] = 0.25
            elif short_setup and position != 1:  # Allow flipping from long to short
                position = -1
                signals[i] = -0.25
            # Exit when price returns to middle (SMA)
            elif position == 1 and close[i] > sma_20[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] < sma_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals