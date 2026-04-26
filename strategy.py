#!/usr/bin/env python3
"""
6h_ADX_Regime_Donchian_20_Breakout_12hTrend_VolumeConfirm_v1
Hypothesis: On 6h timeframe, Donchian(20) breakouts with 12h ADX regime filter (ADX>25 for trend, ADX<20 for range) and volume confirmation produce high-probability trades. In trending regimes (ADX>25), breakouts continue the trend. In ranging regimes (ADX<20), fade at Donchian bands. Uses 12h EMA50 for trend direction filter. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF indicators
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h ADX(14) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM
    tr_period = 14
    tr_12h = pd.Series(tr_12h)
    atr_12h = tr_12h.ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    plus_dm_12h = pd.Series(plus_dm)
    minus_dm_12h = pd.Series(minus_dm)
    plus_dm_smooth = plus_dm_12h.ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    minus_dm_smooth = minus_dm_12h.ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    plus_di_12h = 100 * plus_dm_smooth / atr_12h
    minus_di_12h = 100 * minus_dm_smooth / atr_12h
    
    # DX and ADX
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    dx_12h = pd.Series(dx_12h)
    adx_12h = dx_12h.ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 6h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 6h volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 34 for ADX, 20 for Donchian/volume)
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 12h trend filter (EMA50)
        uptrend_12h = close[i] > ema_50_12h_aligned[i]
        downtrend_12h = close[i] < ema_50_12h_aligned[i]
        
        # 12h ADX regime filter
        adx = adx_12h_aligned[i]
        trending = adx > 25.0
        ranging = adx < 20.0
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Donchian fade conditions (touching bands)
        touch_upper = abs(close[i] - highest_high[i]) / highest_high[i] < 0.001
        touch_lower = abs(close[i] - lowest_low[i]) / lowest_low[i] < 0.001
        
        # Long logic
        long_signal = False
        if trending and uptrend_12h and breakout_up and volume_spike:
            long_signal = True  # Trend breakout continuation
        elif ranging and touch_lower and volume_spike:
            long_signal = True  # Range fade at support
        
        # Short logic
        short_signal = False
        if trending and downtrend_12h and breakout_down and volume_spike:
            short_signal = True  # Trend breakout continuation
        elif ranging and touch_upper and volume_spike:
            short_signal = True  # Range fade at resistance
        
        # Exit conditions
        exit_long = False
        exit_short = False
        if position == 1:
            if trending and not uptrend_12h:  # Trend change
                exit_long = True
            elif ranging and touch_upper:  # Hit resistance in range
                exit_long = True
        elif position == -1:
            if trending and not downtrend_12h:  # Trend change
                exit_short = True
            elif ranging and touch_lower:  # Hit support in range
                exit_short = True
        
        # Update signals and position
        if long_signal and position != 1:
            signals[i] = 0.25
            position = 1
        elif short_signal and position != -1:
            signals[i] = -0.25
            position = -1
        elif exit_long:
            signals[i] = 0.0
            position = 0
        elif exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Regime_Donchian_20_Breakout_12hTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0