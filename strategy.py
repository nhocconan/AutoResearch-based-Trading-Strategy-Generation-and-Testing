#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ADX(14) > 25 for trending regime (bullish if +DI > -DI, bearish if -DI > +DI).
- Donchian channel: 20-period high/low from prior 4h bar.
- Entry: Long when price breaks above upper Donchian AND 1d ADX > 25 AND +DI > -DI AND volume > 1.5 * volume MA(20).
         Short when price breaks below lower Donchian AND 1d ADX > 25 AND -DI > +DI AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below midpoint of Donchian channel,
        exit short when price crosses above midpoint of Donchian channel.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to work in both bull and bear markets via trend filter (ADX) and mean-reversion exits.
Proven pattern from DB: Donchian breakouts with volume and trend filters show strong test performance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) and DI components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Get 4h data for Donchian channel calculation (prior bar OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) from prior bar data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper and lower Donchian bands (20-period)
    upper_donch = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_donch = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    mid_donch = (upper_donch + lower_donch) / 2.0  # Midpoint for exit
    
    # Align HTF indicators to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    di_plus_aligned = align_htf_to_ltf(prices, df_1d, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_1d, di_minus)
    upper_donch_aligned = align_htf_to_ltf(prices, df_4h, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_4h, lower_donch)
    mid_donch_aligned = align_htf_to_ltf(prices, df_4h, mid_donch)
    
    # Calculate volume MA(20) for confirmation (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 14+14+14, 2, 20, 20)  # Need enough bars for ADX, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(di_plus_aligned[i]) or np.isnan(di_minus_aligned[i]) or 
            np.isnan(upper_donch_aligned[i]) or np.isnan(lower_donch_aligned[i]) or np.isnan(mid_donch_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            adx_trending = adx_aligned[i] > 25
            
            # Long: Price breaks above upper Donchian AND ADX trending AND +DI > -DI AND volume confirmed
            if curr_close > upper_donch_aligned[i] and adx_trending and (di_plus_aligned[i] > di_minus_aligned[i]) and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian AND ADX trending AND -DI > +DI AND volume confirmed
            elif curr_close < lower_donch_aligned[i] and adx_trending and (di_minus_aligned[i] > di_plus_aligned[i]) and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below midpoint of Donchian channel (reversion to mean)
            if curr_close < mid_donch_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above midpoint of Donchian channel (reversion to mean)
            if curr_close > mid_donch_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dADX14_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0