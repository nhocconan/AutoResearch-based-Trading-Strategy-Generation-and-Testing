#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout + 12h ADX trend filter + volume confirmation
    # Long: price breaks above Donchian upper AND 12h ADX > 25 AND volume > 1.5x avg
    # Short: price breaks below Donchian lower AND 12h ADX > 25 AND volume > 1.5x avg
    # Exit: price returns to Donchian middle OR ADX < 20 (trend weak)
    # Using 6h timeframe for balance of trade frequency and noise reduction,
    # Donchian for clear breakout levels, ADX for trend strength filter,
    # volume for confirmation of genuine breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h ADX(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_12h index
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilders_smoothing(dx_12h, 14)
    
    # Align 12h ADX to 6h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 6h Donchian(20)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(lookback, n):
        upper[i] = np.max(high[i-lookback:i])
        lower[i] = np.min(low[i-lookback:i])
        middle[i] = (upper[i] + lower[i]) / 2
    
    # 6h volume confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 = strong trend
        strong_trend = adx_12h_aligned[i] > 25
        weak_trend = adx_12h_aligned[i] < 20  # exit condition
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Breakout conditions
        long_breakout = close[i] > upper[i]
        short_breakout = close[i] < lower[i]
        
        # Reversion to middle (exit)
        return_to_middle = (close[i] > lower[i]) and (close[i] < upper[i])
        
        # Entry logic: breakout + trend + volume
        long_entry = long_breakout and strong_trend and vol_confirm
        short_entry = short_breakout and strong_trend and vol_confirm
        
        # Exit logic: return to middle OR weak trend
        long_exit = return_to_middle or weak_trend
        short_exit = return_to_middle or weak_trend
        
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

name = "6h_12h_donchian_adx_volume_v1"
timeframe = "6h"
leverage = 1.0