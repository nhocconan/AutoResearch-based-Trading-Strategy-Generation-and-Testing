#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + Volume + 12h ADX Trend Filter
# Hypothesis: Breakouts of 4h Donchian channels with volume confirmation and 12h ADX trend filter work in both bull and bear markets.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

name = "4h_donchian_breakout_vol_12h_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX for trend strength
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM (Wilder smoothing)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_12h = wilder_smooth(tr, 14)
    plus_di_12h = 100 * wilder_smooth(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilder_smooth(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilder_smooth(dx_12h, 14)
    
    # Align 12h ADX to 4h
    adx_12h_4h = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 4h Donchian channel (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    
    # Volume filter: 4h volume > 20-period average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 19:
            vol_ma_20[i] = np.nan
        else:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx_12h_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        trend_ok = adx_12h_4h[i] > 25
        
        if position == 1:  # Long position
            # Exit: price touches Donchian lower band or trend weakens
            if low[i] <= donch_low[i] or not trend_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches Donchian upper band or trend weakens
            if high[i] >= donch_high[i] or not trend_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of 12h trend with volume confirmation
            if vol_ok and trend_ok:
                if close[i] > donch_high[i]:  # Break above upper band
                    position = 1
                    signals[i] = 0.25
                elif close[i] < donch_low[i]:  # Break below lower band
                    position = -1
                    signals[i] = -0.25
    
    return signals