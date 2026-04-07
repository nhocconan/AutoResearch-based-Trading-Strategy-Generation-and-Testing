#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day volume confirmation and 1-week ADX trend filter.
In strong trends (ADX > 25): breakout entries in trend direction.
In weak trends (ADX <= 25): no trades to avoid whipsaw.
Volume must exceed 20-bar average to confirm breakout authenticity.
Targets 20-40 trades/year by requiring strong trend + volume + breakout confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_volume_1w_adx_v1"
timeframe = "4h"
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
    
    # === 1-WEEK ADX TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    # Calculate ADX on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smoothed_avg(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            result[period-1] = np.nansum(x[:period])
            for i in range(period, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    period = 14
    tr_smooth = smoothed_avg(tr, period)
    dm_plus_smooth = smoothed_avg(dm_plus, period)
    dm_minus_smooth = smoothed_avg(dm_minus, period)
    
    # DI and DX
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed_avg(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # === 1-DAY VOLUME (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    daily_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 4H DONCHIAN CHANNELS (LTF) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation on 4H
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        if np.isnan(adx_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(vol_ma_4h[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        if not strong_trend:
            # No trading in weak/choppy markets
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below midpoint OR trend weakens
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midpoint or adx_aligned[i] <= 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above midpoint OR trend weakens
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midpoint or adx_aligned[i] <= 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation on both 1D and 4H
            if volume[i] <= vol_ma_4h[i] or daily_volume[i] <= vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                continue
            
            # Breakout entries in direction of trend
            if close[i] > highest_high[i]:  # Break above Donchian high
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i]:  # Break below Donchian low
                position = -1
                signals[i] = -0.25
    
    return signals