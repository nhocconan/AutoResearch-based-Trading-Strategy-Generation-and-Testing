#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Confirmation and ADX Trend Filter
Hypothesis: Donchian breakouts capture strong momentum moves. Volume confirms institutional participation.
ADX ensures we only trade in trending markets, avoiding whipsaws in sideways conditions.
Works in bull (breakouts above upper band) and bear (breakdowns below lower band).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for ADX trend filter (once before loop)
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Daily ADX calculation (14 period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        tr_rounded = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        plus_dm_rounded = pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values
        minus_dm_rounded = pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_rounded / tr_rounded
        minus_di = 100 * minus_dm_rounded / tr_rounded
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        
        return adx
    
    adx_daily = calculate_adx(high_daily, low_daily, close_daily, 14)
    adx_daily_aligned = align_htf_to_ltf(prices, df_daily, adx_daily)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20 period)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require above average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(donchian_period, 14) + 14
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx_daily_aligned[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        # ADX filter: only trade when trending (ADX > 25)
        trending = adx_daily_aligned[i] > 25
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian band OR stoploss
            if (close[i] <= donchian_low[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian band OR stoploss
            if (close[i] >= donchian_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
        else:
            # Look for entries: Donchian breakout + volume + trend filter
            long_setup = (close[i] > donchian_high[i] and vol_filter[i] and trending)
            short_setup = (close[i] < donchian_low[i] and vol_filter[i] and trending)
            
            if long_setup:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals