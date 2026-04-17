#!/usr/bin/env python3
"""
12h_Willard_LongTermTrend_Signal
Strategy: Long-term trend following on 12h using Willard's Volatility Breakout with volume confirmation and ADX filter.
Long: Price breaks above Donchian(20) + ADX > 25 + volume > 1.5x average
Short: Price breaks below Donchian(20) + ADX > 25 + volume > 1.5x average
Exit: Price returns to midpoint of Donchian channel or ADX < 20
Position size: 0.25
Designed to capture strong trends with volatility breakouts while avoiding whipsaws in ranging markets.
Timeframe: 12h
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
    
    # Calculate Donchian channel (20-period) on 12h
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    high_series_12h = pd.Series(high_12h)
    low_series_12h = pd.Series(low_12h)
    donchian_high_12h = high_series_12h.rolling(window=20, min_periods=20).max().values
    donchian_low_12h = low_series_12h.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (no shift needed as we're already on 12h)
    donchian_high_aligned = donchian_high_12h
    donchian_low_aligned = donchian_low_12h
    
    # Calculate ADX (14-period) on 1w for trend strength
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period]) / period
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need enough data for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: ADX > 25 for strong trend
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        # Calculate Donchian midpoint for exit
        donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        
        # Entry conditions
        if position == 0:
            # Long: Price breaks above Donchian high + strong trend + volume
            if (close[i] > donchian_high_aligned[i] and strong_trend and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + strong trend + volume
            elif (close[i] < donchian_low_aligned[i] and strong_trend and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to midpoint or trend weakens
            if close[i] < donchian_mid or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to midpoint or trend weakens
            if close[i] > donchian_mid or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Willard_LongTermTrend_Signal"
timeframe = "12h"
leverage = 1.0