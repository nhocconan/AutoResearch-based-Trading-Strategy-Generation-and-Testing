#!/usr/bin/env python3
"""
4h_ADX_VolumeBreakout_Trend
Hypothesis: Combine ADX trend strength with volume breakout signals on 4h timeframe.
Long when ADX > 25, +DI > -DI, and volume breaks above 1.5x 20-period average.
Short when ADX > 25, -DI > +DI, and volume breaks above 1.5x 20-period average.
Uses 1d EMA50 as additional trend filter to avoid counter-trend trades.
Volume confirmation reduces false breakouts. Designed for fewer trades (~25-40/year)
to minimize fee drag while capturing strong trending moves in both bull and bear markets.
"""

name = "4h_ADX_VolumeBreakout_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX calculation (14-period)
    period_adx = 14
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            alpha = 1.0 / period
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smoothed = wilders_smoothing(tr, period_adx)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period_adx)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period_adx)
    
    # DI values
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period_adx)
    
    # Volume SMA (20-period)
    vol_sma20 = np.full(n, np.nan)
    if n >= 20:
        vol_sma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_sma20[i] = (vol_sma20[i-1] * 19 + volume[i]) / 20
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_adx * 2, 20, 50)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or \
           np.isnan(vol_sma20[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume breakout: current volume > 1.5x 20-period average
        volume_breakout = volume[i] > 1.5 * vol_sma20[i]
        
        # Trend filter: price vs 1d EMA50
        uptrend_filter = close[i] > ema50_1d_aligned[i]
        downtrend_filter = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: ADX > 25, +DI > -DI, volume breakout, uptrend filter
            if (adx[i] > 25 and 
                plus_di[i] > minus_di[i] and
                volume_breakout and
                uptrend_filter):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25, -DI > +DI, volume breakout, downtrend filter
            elif (adx[i] > 25 and 
                  minus_di[i] > plus_di[i] and
                  volume_breakout and
                  downtrend_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: ADX weakens (< 20) or trend filter fails
            if (adx[i] < 20 or 
                not uptrend_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: ADX weakens (< 20) or trend filter fails
            if (adx[i] < 20 or 
                not downtrend_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals