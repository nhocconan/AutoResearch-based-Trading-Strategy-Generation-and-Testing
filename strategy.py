#!/usr/bin/env python3
# 12h_1d_adx_volume_breakout_v1
# Strategy: 12h ADX breakout with volume confirmation and 1d EMA trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: ADX > 25 indicates strong trend. Breakouts above/below Donchian channels
# with volume confirmation and aligned with 1d EMA50 trend provide high-probability
# entries in both bull and bear markets. Low-frequency design (15-25 trades/year)
# minimizes fee drag while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_adx_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ADX calculation (14-period)
    # True Range
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # ADX trend strength filter
        strong_trend = adx[i] > 25
        
        # Entry logic: Donchian breakout + volume + ADX + trend alignment
        if (close[i] > donchian_high[i] and  # Break above upper channel
            volume_filter[i] and strong_trend and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < donchian_low[i] and  # Break below lower channel
              volume_filter[i] and strong_trend and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Donchian opposite touch or trend weakening
        elif position == 1 and (close[i] < donchian_low[i] or adx[i] < 20 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high[i] or adx[i] < 20 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals