#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume spike and 1d ADX trend filter
# Uses 12h timeframe to reduce trade frequency and avoid fee drag
# Donchian breakout captures breakouts, volume spike confirms strength, 1d ADX > 25 filters for trending markets
# Works in both bull and bear markets by only trading in direction of daily trend as measured by ADX
# Target: 15-25 trades/year per symbol (60-100 total) to stay within fee limits

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX on daily data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    def smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period]) if period > 1 else arr[0]
        # Subsequent values using Wilder's smoothing
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth(tr, 14)
    plus_di = 100 * smooth(plus_dm, 14) / atr
    minus_di = 100 * smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth(dx, 14)
    
    # Calculate 20-period Donchian channels on 12h data
    def donchian_channel(arr, period):
        upper = np.full_like(arr, np.nan)
        lower = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < period - 1:
                continue
            upper[i] = np.max(arr[i-period+1:i+1])
            lower[i] = np.min(arr[i-period+1:i+1])
        return upper, lower
    
    upper_channel, lower_channel = donchian_channel(high, 20)
    lower_channel_exit, upper_channel_exit = donchian_channel(low, 20)  # For exit signals
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 12-hour timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel) if len(upper_channel) == len(df_1d) else upper_channel
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel) if len(lower_channel) == len(df_1d) else lower_channel
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian + volume spike + strong trend (ADX > 25)
            if (close[i] > upper_channel_aligned[i] and vol_spike[i] and adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + volume spike + strong trend (ADX > 25)
            elif (close[i] < lower_channel_aligned[i] and vol_spike[i] and adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < lower_channel_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_channel_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_1dADX25_Trend"
timeframe = "12h"
leverage = 1.0