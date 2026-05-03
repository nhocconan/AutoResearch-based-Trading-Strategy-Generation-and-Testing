#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w volume confirmation and 1d ADX trend filter
# Donchian breakout captures sustained momentum, 1w volume confirms institutional participation,
# 1d ADX > 25 ensures we only trade in strong trends to avoid whipsaws in ranging markets.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag and work in both bull/bear markets.

name = "12h_Donchian20_1wVolume_ADXTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w volume spike (volume > 1.5 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1w = df_1w['volume'].values > (1.5 * vol_ema_20)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilder_smooth(tr, 14)
    plus_di_14 = 100 * wilder_smooth(plus_dm, 14) / atr_14
    minus_di_14 = 100 * wilder_smooth(minus_dm, 14) / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = wilder_smooth(dx, 14)
    
    # Align HTF indicators to 12h timeframe
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Increased warmup for ADX calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike_1w_aligned[i]) or np.isnan(adx_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        is_strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Donchian breakout above upper band + 1w volume spike + strong trend
            if high[i] > highest_high[i] and volume_spike_1w_aligned[i] and is_strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band + 1w volume spike + strong trend
            elif low[i] < lowest_low[i] and volume_spike_1w_aligned[i] and is_strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakdown below middle band OR reverse signal
            middle = (highest_high[i] + lowest_low[i]) / 2
            if low[i] < middle or (low[i] < lowest_low[i] and volume_spike_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout above middle band OR reverse signal
            middle = (highest_high[i] + lowest_low[i]) / 2
            if high[i] > middle or (high[i] > highest_high[i] and volume_spike_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals