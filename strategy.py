#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h ADX regime filter and volume confirmation.
- Primary timeframe: 6h for execution.
- Donchian breakout: Long when close > highest high of past 20 bars, Short when close < lowest low of past 20 bars.
- Regime filter: 12h ADX(14) > 25 indicates trending market (favor breakouts), ADX < 20 indicates ranging (avoid breakouts).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying upward breakouts in uptrend, in bear via selling downward breakouts in downtrend.
- Uses ADX to avoid false breakouts in ranging markets, improving win rate.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def adx(high, low, close, period=14):
    """Average Directional Index (ADX)"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ and DM-
    def smm(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.nanmean(arr[:period])
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + (arr[i] if not np.isnan(arr[i]) else 0)) / period
        return result
    
    atr = smm(tr, period)
    dm_plus_smooth = smm(dm_plus, period)
    dm_minus_smooth = smm(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_result = smm(dx, period)
    
    return adx_result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h ADX(14) for regime filter
    adx_12h = adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 30)  # Donchian + volume MA + ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Donchian breakout with regime and volume filters
            if adx_12h_aligned[i] > 25 and volume_spike[i]:  # Trending market
                if close[i] > highest_high[i]:  # Upward breakout
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low[i]:  # Downward breakout
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below Donchian middle or opposite signal
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian middle or opposite signal
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > donchian_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0