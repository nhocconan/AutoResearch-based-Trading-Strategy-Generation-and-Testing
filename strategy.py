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
    
    # Get daily data for volatility and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily ATR(14) for volatility
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily ADX(14) for trend strength
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12-hour Donchian channels (15-period)
    high_15 = pd.Series(high).rolling(window=15, min_periods=15).max().values
    low_15 = pd.Series(low).rolling(window=15, min_periods=15).min().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(high_15[i]) or 
            np.isnan(low_15[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above SMA50 for long, below for short
        sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        uptrend = close[i] > sma_50[i]
        downtrend = close[i] < sma_50[i]
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_15[i-1]  # Break above previous high
        breakout_down = close[i] < low_15[i-1]  # Break below previous low
        
        # Volatility filter: ensure sufficient ATR
        atr_ma = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
        atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
        if np.isnan(atr_ma_aligned[i]):
            signals[i] = 0.0
            continue
        vol_ok = atr_aligned[i] > (atr_ma_aligned[i] * 0.15)
        
        # Long conditions: uptrend + strong trend + breakout up + volatility
        long_condition = uptrend and strong_trend and breakout_up and vol_ok
        
        # Short conditions: downtrend + strong trend + breakout down + volatility
        short_condition = downtrend and strong_trend and breakout_down and vol_ok
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Donchian breakout
        elif position == 1 and close[i] < low_15[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > high_15[i-1]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian15_Breakout_ADX25_SMA50_Trend"
timeframe = "12h"
leverage = 1.0