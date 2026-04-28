# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d/1w confluence using Keltner Channel breakout + ADX trend filter.
Keltner Channel (20, 1.5) captures volatility-based breakouts. ADX(14) > 25 filters for trending markets.
Includes volume confirmation and session filter (8-20 UTC) to avoid low-liquidity periods.
Designed to work in both bull and bear markets by requiring strong trend + volume confirmation.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # Get daily data for Keltner Channel and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily high/low/close for Keltner Channel
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Keltner Channel components (20, 1.5)
    # Middle line = EMA(20)
    close_1d_series = pd.Series(close_1d)
    kelter_mid = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR for channel width
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    kelter_upper = kelter_mid + 1.5 * atr
    kelter_lower = kelter_mid - 1.5 * atr
    
    # ADX calculation (14)
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Weekly trend filter: price above/below weekly EMA20
    close_1w_series = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    kelter_upper_aligned = align_htf_to_ltf(prices, df_1d, kelter_upper)
    kelter_lower_aligned = align_htf_to_ltf(prices, df_1d, kelter_lower)
    kelter_mid_aligned = align_htf_to_ltf(prices, df_1d, kelter_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kelter_upper_aligned[i]) or np.isnan(kelter_lower_aligned[i]) or 
            np.isnan(kelter_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Weekly trend filter: price above/below weekly EMA20
        weekly_uptrend = close[i] > ema20_1w_aligned[i]
        weekly_downtrend = close[i] < ema20_1w_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above Keltner Upper with volume, strong trend, and weekly uptrend
        # Short: price breaks below Keltner Lower with volume, strong trend, and weekly downtrend
        long_entry = (close[i] > kelter_upper_aligned[i]) and vol_filter and strong_trend and weekly_uptrend
        short_entry = (close[i] < kelter_lower_aligned[i]) and vol_filter and strong_trend and weekly_downtrend
        
        # Exit conditions: price returns to Keltner middle line
        long_exit = (close[i] < kelter_mid_aligned[i])
        short_exit = (close[i] > kelter_mid_aligned[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Keltner_Breakout_ADX25_WeeklyEMA20_Volume_Session"
timeframe = "6h"
leverage = 1.0