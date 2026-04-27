#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian(20) breakout with 12-hour volume confirmation and 12-hour ADX trend filter.
Enters long when price breaks above Donchian upper band with above-average volume and ADX > 25 (trending).
Enters short when price breaks below Donchian lower band with above-average volume and ADX > 25.
Uses 12h timeframe for volume and ADX to reduce noise while maintaining 4h execution for timely entries.
Donchian breakouts capture breakouts with clear risk management, volume confirms conviction, ADX filters ranging markets.
Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag and avoid overtrading.
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
    
    # Calculate 4h Donchian(20) channels
    # Upper band: 20-period high
    # Lower band: 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for volume and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h volume MA(20)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 12h ADX(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        """Wilder's smoothing (like EMA with alpha=1/period)"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    tr_smoothed = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, 14)
    
    # Align 12h indicators to 4h timeframe
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian bands, volume MA, and ADX
    start_idx = max(20, 20, 14+14+14)  # Donchian(20), Vol MA(20), ADX(14+14+14 for smoothing)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        adx_now = adx_aligned[i]
        
        # Volume filter: volume > 1.5x 12h average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # ADX filter: trending market (ADX > 25)
        trend_filter = adx_now > 25
        
        # Donchian breakout signals
        breakout_up = price_now > donchian_upper[i-1]  # break above previous upper band
        breakout_down = price_now < donchian_lower[i-1]  # break below previous lower band
        
        # Entry conditions
        if position == 0:
            # Long: breakout above upper band with volume + trend
            if breakout_up and vol_filter and trend_filter:
                signals[i] = size
                position = 1
            # Short: breakout below lower band with volume + trend
            elif breakout_down and vol_filter and trend_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower band or ADX weakens (< 20)
            if price_now < donchian_lower[i] or adx_now < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above upper band or ADX weakens (< 20)
            if price_now > donchian_upper[i] or adx_now < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DonchianBreakout_12hVolume_ADXTrend"
timeframe = "4h"
leverage = 1.0