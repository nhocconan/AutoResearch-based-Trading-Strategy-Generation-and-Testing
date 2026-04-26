#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_ADXFilter_VolumeConfirm_v1
Hypothesis: Donchian(20) breakout on 6h with 1d EMA50 trend filter, ADX>25 for trend strength, and volume confirmation captures strong directional moves while avoiding chop. Works in bull/bear by only taking breakouts in direction of 1d trend. Targets 12-30 trades/year via tight entry conditions (trend + breakout + volume + ADX). Uses discrete sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d ADX(14) for trend strength filter
    # Calculate ADX components: +DM, -DM, TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
        return result
    
    period_adx = 14
    tr_smooth = wilder_smooth(tr, period_adx)
    plus_dm_smooth = wilder_smooth(plus_dm, period_adx)
    minus_dm_smooth = wilder_smooth(minus_dm, period_adx)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
    minus_di = 100 * minus_dm_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = wilder_smooth(dx, period_adx)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian(20) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Donchian (20), EMA50 (50), ADX (14+14=28), volume MA (20)
    start_idx = max(lookback, 50, 28, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        trend_val = ema50_1d_aligned[i]
        adx_val = adx_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(highest_high_val) or np.isnan(lowest_low_val) or 
            np.isnan(trend_val) or np.isnan(adx_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Trend strength filter: ADX > 25
        strong_trend = adx_val > 25
        
        # Entry conditions: Donchian breakout in direction of trend + ADX > 25 + volume
        long_condition = (close_val > highest_high_val) and is_uptrend and strong_trend and vol_conf
        short_condition = (close_val < lowest_low_val) and is_downtrend and strong_trend and vol_conf
        
        # Exit conditions: opposite Donchian touch or trend reversal or ADX < 20
        long_exit = (position == 1 and 
                    (close_val < lowest_low_val or not is_uptrend or adx_val < 20))
        short_exit = (position == -1 and 
                     (close_val > highest_high_val or not is_downtrend or adx_val < 20))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_ADXFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0