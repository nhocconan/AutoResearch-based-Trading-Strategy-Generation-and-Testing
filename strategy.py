#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and ADX Trend Filter
Enters long when price breaks above weekly Donchian upper channel with volume > 1.5x daily average and ADX > 25
Enters short when price breaks below weekly Donchian lower channel with volume > 1.5x daily average and ADX > 25
Exits when price returns to the weekly median line or volume drops below average.
Designed for 1d timeframe to capture major trend continuation moves with institutional volume.
Target: 10-25 trades/year (40-100 total over 4 years) by requiring confluence of breakout, volume, and trend.
Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend.
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
    
    # === Weekly Donchian Channel (20-period) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    median_20 = (upper_20 + lower_20) / 2
    
    # Align to daily timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    median_20_aligned = align_htf_to_ltf(prices, df_1w, median_20)
    
    # === Daily Volume Confirmation (1.5x 20-day average) ===
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Weekly ADX Trend Filter (ADX > 25) ===
    # Calculate True Range (TR)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate Directional Movement (+DM, -DM)
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr_1w = wilders_smoothing(tr, period)
    plus_di_1w = np.where(atr_1w != 0, 100 * wilders_smoothing(plus_dm, period) / atr_1w, 0)
    minus_di_1w = np.where(atr_1w != 0, 100 * wilders_smoothing(minus_dm, period) / atr_1w, 0)
    dx_1w = np.where((plus_di_1w + minus_di_1w) != 0, 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w), 0)
    adx_1w = wilders_smoothing(dx_1w, period)
    
    # Handle division by zero
    adx_1w = np.where((plus_di_1w + minus_di_1w) == 0, 0, adx_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for all calculations
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(median_20_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > volume_ma_20[i] * 1.5
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx_1w_aligned[i] > 25
        
        # Breakout conditions
        breakout_up = high[i] > upper_20_aligned[i]   # Price breaks above weekly upper channel
        breakdown_down = low[i] < lower_20_aligned[i] # Price breaks below weekly lower channel
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish breakout above weekly upper channel with volume confirmation and trend
            if breakout_up and vol_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish breakdown below weekly lower channel with volume confirmation and trend
            elif breakdown_down and vol_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to weekly median line or volume fails
        elif position == 1:
            # Exit long: price returns to weekly median line or volume confirmation fails
            if low[i] <= median_20_aligned[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly median line or volume confirmation fails
            if high[i] >= median_20_aligned[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_VolumeConfirmation_ADXFilter"
timeframe = "1d"
leverage = 1.0