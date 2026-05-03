#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX25 trend filter and 1w volume confirmation
# Donchian breakouts capture institutional participation at key levels. 1d ADX>25 ensures
# we only trade in trending markets, avoiding whipsaws in ranging conditions. 1w volume
# spike confirms institutional commitment. Designed for 50-150 total trades over 4 years
# (12-37/year) to minimize fee drag. Works in bull markets via breakout continuation
# and in bear markets via breakdown shorts with trend filter.

name = "6h_Donchian20_1dADX25_1wVolumeSpike"
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
    
    # Get 1d data for ADX25 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) with smoothing period 25
    # ADX requires +DI, -DI, and TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d bars
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    plus_di_1d = 100 * wilder_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilder_smooth(dx_1d, 14)  # ADX is smoothed DX
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w volume EMA(20)
    vol_1w = df_1w['volume'].values
    vol_ema_20_1w = pd.Series(vol_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ema_20_1w)
    
    # Calculate 6h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have Donchian data
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ema_20_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5 * 1w volume EMA(20)
        volume_spike = volume[i] > (1.5 * vol_ema_20_1w_aligned[i])
        
        if position == 0:
            # Long: price breaks above Donchian high in 1d uptrend (ADX>25) with volume spike
            if close[i] > donchian_high[i] and adx_1d_aligned[i] > 25 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in 1d downtrend (ADX>25) with volume spike
            elif close[i] < donchian_low[i] and adx_1d_aligned[i] > 25 and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or loses 1d trend (ADX<20)
            if close[i] < donchian_low[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or loses 1d trend (ADX<20)
            if close[i] > donchian_high[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals