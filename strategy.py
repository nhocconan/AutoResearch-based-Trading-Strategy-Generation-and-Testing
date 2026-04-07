#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Volume and ADX Trend Filter
# Hypothesis: Breakouts of Donchian(20) channels on 12h timeframe, filtered by ADX>25 for trend strength
# and volume confirmation, work in both bull and bear markets by capturing strong directional moves.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_donchian20_volume_adx_v14"
timeframe = "12h"
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
    
    # Get daily data for Donchian channels and ADX
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # 20-period rolling high and low
    donchian_high = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h
    donchian_high_12h = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_daily, donchian_low)
    
    # Calculate ADX (14-period) for trend strength
    # True Range
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    
    # Directional Movement
    up_move = high_daily - np.roll(high_daily, 1)
    down_move = np.roll(low_daily, 1) - low_daily
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 12h
    adx_12h = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume filter: 12h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(adx_12h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_ok = adx_12h[i] > 25
        
        if position == 1:  # Long position
            # Exit: price touches Donchian low or trend weakens
            if low[i] <= donchian_low_12h[i] or not trend_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches Donchian high or trend weakens
            if high[i] >= donchian_high_12h[i] or not trend_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout entry with volume and trend confirmation
            if vol_ok and trend_ok:
                # Long breakout: price breaks above Donchian high
                if high[i] > donchian_high_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price breaks below Donchian low
                elif low[i] < donchian_low_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals