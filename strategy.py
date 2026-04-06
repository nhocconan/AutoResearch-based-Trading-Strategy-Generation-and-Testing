#!/usr/bin/env python3
"""
6h Donchian(20) breakout with 12h ADX trend filter and volume confirmation
Hypothesis: 6h Donchian breakouts capture intermediate-term momentum. Filter by 12h ADX > 25 for trending markets and volume confirmation for conviction. Works in bull (buy breakouts above 12h ADX) and bear (sell breakdowns below 12h ADX). Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get 12h data for trend filter (ADX)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX components on 12h data
    # True Range
    tr_12h = np.maximum(
        high_12h[1:] - low_12h[1:],
        np.abs(high_12h[1:] - close_12h[:-1]),
        np.abs(low_12h[1:] - close_12h[:-1])
    )
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (14-period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_14 = wilder_smooth(np.concatenate([[np.nan], tr_12h]), 14)
    dm_plus_14 = wilder_smooth(np.concatenate([[np.nan], dm_plus]), 14)
    dm_minus_14 = wilder_smooth(np.concatenate([[np.nan], dm_minus]), 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(tr_14 != 0, dm_plus_14 / tr_14 * 100, 0)
    di_minus = np.where(tr_14 != 0, dm_minus_14 / tr_14 * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilder_smooth(np.concatenate([[np.nan], dx[1:]]), 14)
    
    # 12h trend strength: ADX > 25 indicates trending market
    trend_12h = np.where(adx > 25, 1, 0)  # 1 = trending, 0 = ranging
    
    # Align 12h trend to 6h timeframe
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Get 12h data for volume confirmation
    volume_12h = df_12h['volume'].values
    
    # 20-period average volume on 12h
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Donchian channels (20-period) from 6h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 40  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_12h_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 12h average volume (scaled)
        # Scale 12h volume to 6h: approx 1/2 of 12h volume (since 2x 6h in 12h)
        vol_threshold = vol_ma_12h_aligned[i] / 2.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against trend filter (ranging market)
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                trend_12h_aligned[i] == 0 or  # Exit when market becomes ranging
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against trend filter (ranging market)
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                trend_12h_aligned[i] == 0 or  # Exit when market becomes ranging
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 6 bars flat
            if bars_since_entry >= 6:
                # Breakout entries: upper/lower with trend filter
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper with trending 12h market + volume
                if bull_breakout and trend_12h_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with trending 12h market + volume
                elif bear_breakout and trend_12h_aligned[i] == 1 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals