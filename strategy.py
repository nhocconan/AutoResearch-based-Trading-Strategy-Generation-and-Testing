#!/usr/bin/env python3
"""
1d ADX + 30-period Donchian Breakout with Volume Spike
Trend-following strategy for daily timeframe using ADX trend strength filter
and Donchian channel breakouts. Uses weekly trend filter for multi-timeframe
confirmation. Designed for low trade frequency with strong trend capture.
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
    
    # Calculate ADX (14-period) for trend strength
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    period_adx = 14
    tr_smooth = smooth_wilder(tr, period_adx)
    plus_dm_smooth = smooth_wilder(plus_dm, period_adx)
    minus_dm_smooth = smooth_wilder(minus_dm, period_adx)
    
    # DI values
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, period_adx)
    
    # 30-period Donchian channels
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donch_high = rolling_max(high, 30)
    donch_low = rolling_min(low, 30)
    
    # Volume spike (2x 20-period average)
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (2.0 * vol_ma)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema_50_1w = np.full_like(close_1w, np.nan)
    for i in range(49, len(close_1w)):
        if i == 49:
            ema_50_1w[i] = np.mean(close_1w[:50])
        else:
            ema_50_1w[i] = (close_1w[i] * 2/51) + (ema_50_1w[i-1] * 49/51)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 60  # need enough history for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx[i]
        upper_break = price > donch_high[i-1]  # break above previous period's high
        lower_break = price < donch_low[i-1]   # break below previous period's low
        weekly_trend_up = price > ema_50_1w_aligned[i]
        weekly_trend_down = price < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: ADX > 25 (strong trend) + break above Donchian high + volume spike + weekly uptrend
            if (adx_val > 25 and upper_break and volume_spike[i] and weekly_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 + break below Donchian low + volume spike + weekly downtrend
            elif (adx_val > 25 and lower_break and volume_spike[i] and weekly_trend_down):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: hold until break below Donchian low or ADX weakens
            signals[i] = 0.25
            if price < donch_low[i-1] or adx_val < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until break above Donchian high or ADX weakens
            signals[i] = -0.25
            if price > donch_high[i-1] or adx_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_ADX_Donchian30_VolumeSpike_WeeklyEMA50"
timeframe = "1d"
leverage = 1.0