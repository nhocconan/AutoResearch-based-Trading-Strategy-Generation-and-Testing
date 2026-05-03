#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX(14) trend strength + 12h Donchian(20) breakout + volume confirmation
# ADX > 25 identifies strong trending markets (works in both bull/bear).
# Donchian breakout in direction of 12h trend captures momentum with filtering.
# Volume spike confirms conviction. Designed for 12-30 trades/year on 6h to minimize fee drag.
# Uses discrete position sizing (0.0, ±0.25) to reduce churn.

name = "6h_ADX_Trend_DonchianBreakout_12hVolume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC) to avoid datetime64 issues
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for HTF indicators (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX(14) for trend strength filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_12h = wilder_smooth(tr, 14)
    plus_di_12h = 100 * wilder_smooth(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilder_smooth(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilder_smooth(dx_12h, 14)
    
    # Align ADX to 6h timeframe (completed 12h bar only)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 12h Donchian channels (20-period)
    high_max_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    high_max_12h_aligned = align_htf_to_ltf(prices, df_12h, high_max_12h)
    low_min_12h_aligned = align_htf_to_ltf(prices, df_12h, low_min_12h)
    
    # Calculate 6h volume EMA(20) for confirmation
    volume_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(high_max_12h_aligned[i]) or 
            np.isnan(low_min_12h_aligned[i]) or np.isnan(volume_ema_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX trend strength filter: > 25 indicates strong trend
        strong_trend = adx_12h_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5 * 20-period EMA
        volume_spike = volume[i] > (1.5 * volume_ema_20[i])
        
        # Donchian breakout conditions
        donchian_breakout_up = close[i] > high_max_12h_aligned[i]
        donchian_breakout_down = close[i] < low_min_12h_aligned[i]
        
        if position == 0:
            # Enter long: strong uptrend + Donchian breakout up + volume spike
            if strong_trend and donchian_breakout_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: strong trend + Donchian breakout down + volume spike
            elif strong_trend and donchian_breakout_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian breakout down or loss of trend strength
            if donchian_breakout_down or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian breakout up or loss of trend strength
            if donchian_breakout_up or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals