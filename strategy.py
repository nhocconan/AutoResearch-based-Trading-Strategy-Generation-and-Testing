#!/usr/bin/env python3
"""
1d_Weekly_Donchian_Trend_Filter
Hypothesis: Uses weekly Donchian channel breakouts on daily timeframe with volume confirmation and ADX trend filter.
Enters long when price breaks above weekly upper band with ADX>25, short when breaks below weekly lower band with ADX>25.
Designed for low trade frequency (~10-20/year) to capture major trends while avoiding whipsaws in ranging markets.
Works in both bull (breakouts) and bear (breakdowns) markets by following the weekly trend.
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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    donchian_high = np.full(len(weekly_high), np.nan)
    donchian_low = np.full(len(weekly_low), np.nan)
    
    for i in range(20, len(weekly_high)):
        donchian_high[i] = np.max(weekly_high[i-20:i])
        donchian_low[i] = np.min(weekly_low[i-20:i])
    
    # Align to daily timeframe (waits for weekly bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # ADX filter on weekly timeframe (trend strength)
    # Calculate True Range components
    tr1 = weekly_high[1:] - weekly_low[1:]
    tr2 = np.abs(weekly_high[1:] - weekly_close[:-1]) if len(weekly_high) > 1 else np.array([])
    tr3 = np.abs(weekly_low[1:] - weekly_close[:-1]) if len(weekly_high) > 1 else np.array([])
    
    weekly_close = df_weekly['close'].values
    if len(weekly_high) > 1:
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    else:
        tr = np.array([np.nan])
    
    # Calculate ATR (14-period)
    atr_period = 14
    atr = np.full(len(tr), np.nan)
    for i in range(atr_period, len(tr)):
        if i == atr_period:
            atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate +DM and -DM
    up_move = np.diff(weekly_high)
    down_move = -np.diff(weekly_low)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: smoothed = prev * (1-1/period) + current * (1/period)
        alpha = 1.0 / period
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] * (1 - alpha) + arr[i] * alpha
        return result
    
    plus_di = 100 * wilder_smooth(plus_dm, atr_period) / atr
    minus_di = 100 * wilder_smooth(minus_dm, atr_period) / atr
    
    # Calculate DX and ADX
    dx = np.full(len(plus_di), np.nan)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = wilder_smooth(dx, atr_period)
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Volume confirmation: daily volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly upper band with strong trend and volume spike
            if close[i] > donchian_high_aligned[i] and adx_aligned[i] > 25 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly lower band with strong trend and volume spike
            elif close[i] < donchian_low_aligned[i] and adx_aligned[i] > 25 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below weekly lower band or trend weakens
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above weekly upper band or trend weakens
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Trend_Filter"
timeframe = "1d"
leverage = 1.0