#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above Donchian(20) high with bullish 1d trend (ADX>25 and +DI>-DI).
Short when price breaks below Donchian(20) low with bearish 1d trend (ADX>25 and +DI<+DI).
Exit when price crosses 10-period EMA.
Uses volume spike (2x average) to avoid false breakouts.
Designed for low trade frequency (20-40/year) to minimize fee drag.
Works in bull markets via trend-following breakouts and in bear markets via short breakdowns.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])  # Skip index 0
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    tr_smooth = smooth_wilder(tr, period)
    plus_dm_smooth = smooth_wilder(plus_dm, period)
    minus_dm_smooth = smooth_wilder(minus_dm, period)
    
    # DI values
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = smooth_wilder(dx, period)
    
    # Align ADX, +DI, -DI to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h EMA(10) for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian lookback
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema10[i]) or np.isnan(vol_avg_20[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with bullish 1d trend and volume spike
            if (close[i] > donchian_high[i] and 
                adx_aligned[i] > 25 and 
                plus_di_aligned[i] > minus_di_aligned[i] and
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with bearish 1d trend and volume spike
            elif (close[i] < donchian_low[i] and 
                  adx_aligned[i] > 25 and 
                  plus_di_aligned[i] < minus_di_aligned[i] and
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses 10-period EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below EMA10
                if close[i] < ema10[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above EMA10
                if close[i] > ema10[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_20_1dADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0
#%%