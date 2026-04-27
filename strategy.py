#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian channel breakout with volume confirmation and ADX trend filter.
# Donchian breakouts capture momentum moves; volume confirms institutional participation.
# ADX > 25 ensures we only trade in trending markets, reducing whipsaws in ranging conditions.
# Designed for moderate trade frequency (target: 80-150 total trades over 4 years) to balance opportunity and cost.
# Works in bull markets (upward breakouts) and bear markets (downward breakouts).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation (same timeframe, but we still use the helper for consistency)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian channels (20-period)
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Calculate ADX for trend filtering
    # Calculate True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3 = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = high_4h - np.concatenate([[high_4h[0]], high_4h[:-1]])
    down_move = np.concatenate([[low_4h[0]], low_4h[:-1]]) - low_4h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (14-period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above upper Donchian with ADX > 25 and volume
        if (close[i] > upper_20_aligned[i] and 
            adx_aligned[i] > 25 and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short breakdown: price breaks below lower Donchian with ADX > 25 and volume
        elif (close[i] < lower_20_aligned[i] and 
              adx_aligned[i] > 25 and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signal or ADX drops below 20
        elif position == 1 and (close[i] < lower_20_aligned[i] or adx_aligned[i] < 20):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > upper_20_aligned[i] or adx_aligned[i] < 20):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_ADX25_VolumeFilter"
timeframe = "4h"
leverage = 1.0