#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + volume confirmation + ADX regime filter.
    # In trending markets (ADX > 25), trade breakouts in direction of trend.
    # In ranging markets (ADX < 20), fade Donchian band touches.
    # Volume filter ensures participation. Target: 75-200 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
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
    
    # Wilder's smoothing function
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d > 0, (dm_plus_smooth / atr_1d) * 100, 0)
    di_minus = np.where(atr_1d > 0, (dm_minus_smooth / atr_1d) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 4h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 20-period MA
        volume_filter = volume[i] > volume_ma[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Using previous bar's channel
        breakout_down = close[i] < lowest_low[i-1]
        touch_upper = abs(high[i] - highest_high[i-1]) < 0.001 * high[i]  # Near upper band
        touch_lower = abs(low[i] - lowest_low[i-1]) < 0.001 * low[i]    # Near lower band
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        if is_trending and volume_filter:
            # In trending market: trade breakouts in direction of trend
            # Use 4h EMA50 to determine trend direction
            if i >= 50:
                ema50 = pd.Series(close[:i+1]).ewm(span=50, adjust=False, min_periods=50).mean().iloc[-1]
                uptrend = close[i] > ema50
                downtrend = close[i] < ema50
                
                long_entry = breakout_up and uptrend
                short_entry = breakout_down and downtrend
        elif is_ranging and volume_filter:
            # In ranging market: fade Donchian band touches
            long_entry = touch_lower
            short_entry = touch_upper
        
        # Exit conditions: opposite signal or middle band reversion
        long_exit = False
        short_exit = False
        
        if is_trending:
            # Exit on opposite breakout or close below/above EMA50
            if i >= 50:
                ema50 = pd.Series(close[:i+1]).ewm(span=50, adjust=False, min_periods=50).mean().iloc[-1]
                long_exit = breakout_down or close[i] < ema50
                short_exit = breakout_up or close[i] > ema50
        else:  # ranging
            # Exit when price reverts to middle of channel
            middle = (highest_high[i-1] + lowest_low[i-1]) / 2
            long_exit = close[i] > middle
            short_exit = close[i] < middle
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_adx_volume_v1"
timeframe = "4h"
leverage = 1.0