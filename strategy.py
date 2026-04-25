#!/usr/bin/env python3
"""
6h_WeeklyDonchian20_Breakout_1dADX25_TrendFilter
Hypothesis: Weekly Donchian(20) breakouts aligned with daily ADX>25 trend capture strong momentum moves.
Works in bull/bear via daily ADX trend filter (only trade in trend direction). Uses 6h timeframe for lower frequency.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX) with min_periods"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = np.nan
    dm_minus[0] = np.nan
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            else:
                result[i] = result[i-1]
        return result
    
    tr_smooth = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # Directional Indicators
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period)
    
    return adx

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel with min_periods"""
    if len(high) < period:
        return np.full_like(high, np.nan), np.full_like(low, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian(20) breakout (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Donchian(20) levels
    donch_upper, donch_lower = calculate_donchian(df_1w['high'].values, df_1w['low'].values, 20)
    donch_upper_aligned = align_htf_to_ltf(prices, df_1w, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1w, donch_lower)
    
    # Daily data for ADX>25 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily ADX
    adx = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly Donchian (20) + daily ADX (14+14) + volume MA (20)
    start_idx = max(20, 28, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals - require: Weekly Donchian breakout + ADX>25 + volume spike
            long_breakout = curr_close > donch_upper_aligned[i]
            short_breakout = curr_close < donch_lower_aligned[i]
            strong_trend = adx_aligned[i] > 25
            vol_confirmed = volume_spike[i] and vol_ma[i] > 0
            
            if long_breakout and strong_trend and vol_confirmed:
                signals[i] = 0.25
                position = 1
            elif short_breakout and strong_trend and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below weekly Donchian lower or ADX weakens
            if curr_close < donch_lower_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above weekly Donchian upper or ADX weakens
            if curr_close > donch_upper_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_1dADX25_TrendFilter"
timeframe = "6h"
leverage = 1.0