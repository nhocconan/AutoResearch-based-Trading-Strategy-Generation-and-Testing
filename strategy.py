#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ADX(14)>25 trend filter and 1d volume confirmation
# Donchian breakouts capture strong momentum moves. ADX filter ensures we only trade in trending regimes,
# avoiding whipsaws in ranging markets. Volume confirmation from 1d ensures institutional participation.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets via breakout longs and in bear markets via breakdown shorts with trend filter.

name = "6h_Donchian20_12hADX25_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for ADX(14) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on 12h
    df_12h_high = df_12h['high'].values
    df_12h_low = df_12h['low'].values
    df_12h_close = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(df_12h_high[1:] - df_12h_low[1:])
    tr2 = np.abs(df_12h_high[1:] - df_12h_close[:-1])
    tr3 = np.abs(df_12h_low[1:] - df_12h_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((df_12h_high[1:] - df_12h_high[:-1]) > (df_12h_low[:-1] - df_12h_low[1:]),
                       np.maximum(df_12h_high[1:] - df_12h_high[:-1], 0), 0)
    dm_minus = np.where((df_12h_low[:-1] - df_12h_low[1:]) > (df_12h_high[1:] - df_12h_high[:-1]),
                        np.maximum(df_12h_low[:-1] - df_12h_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ , DM- using Wilder's smoothing (alpha = 1/14)
    def wilder_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(x[1:period])  # skip first NaN in tr
            for i in range(period, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    atr = wilder_smoothing(tr, 14)
    dm_plus_smooth = wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    for i in range(14, len(dx)):
        if np.isnan(adx[i-1]):
            adx[i] = np.nanmean(dx[14:i+1]) if i >= 14 else np.nan
        else:
            adx[i] = adx[i-1] - (adx[i-1] / 14) + dx[i]
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 1d volume
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    
    # Calculate Donchian(20) on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have Donchian data
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ema_20_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5 * 20-period EMA on 1d volume
        volume_spike = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: price breaks above Donchian upper band in 12h uptrend (ADX>25) with volume spike
            if close[i] > highest_20[i] and adx_aligned[i] > 25 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band in 12h downtrend (ADX>25) with volume spike
            elif close[i] < lowest_20[i] and adx_aligned[i] > 25 and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower band or ADX drops below 20
            if close[i] < lowest_20[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper band or ADX drops below 20
            if close[i] > highest_20[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals