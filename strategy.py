#!/usr/bin/env python3
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
    
    # Get 1d data for primary trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w ADX(14) for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # first value
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx, additional_delay_bars=0)
    
    # Get 12h data for entry timing
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) for entry
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    upper = np.full(len(high_12h), np.nan)
    lower = np.full(len(high_12h), np.nan)
    for i in range(20, len(high_12h)):
        upper[i] = np.max(high_12h[i-20:i])
        lower[i] = np.min(low_12h[i-20:i])
    
    donch_upper_12h = upper
    donch_lower_12h = lower
    donch_upper_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_upper_12h)
    donch_lower_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_lower_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need EMA and Donchian
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donch_upper_12h_aligned[i]) or 
            np.isnan(donch_lower_12h_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema = ema_34_1d_aligned[i]
        upper = donch_upper_12h_aligned[i]
        lower = donch_lower_12h_aligned[i]
        adx_val = adx_aligned[i]
        
        # Regime filter: only trade when ADX > 25 (trending market)
        if adx_val < 25:
            signals[i] = 0.0
            continue
        
        # Entry conditions: breakout with trend filter
        if position == 0:
            # Long: break above upper band + price above EMA
            if close[i] > upper and close[i] > ema:
                signals[i] = size
                position = 1
            # Short: break below lower band + price below EMA
            elif close[i] < lower and close[i] < ema:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below EMA or below lower band
            if close[i] < ema or close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above EMA or above upper band
            if close[i] > ema or close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_EMA34_Donchian20_TrendFilter_1wADX"
timeframe = "12h"
leverage = 1.0