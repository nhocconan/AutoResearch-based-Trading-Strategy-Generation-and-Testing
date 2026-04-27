#!/usr/bin/env python3
"""
12h Keltner Channel Breakout with 1-week trend filter and 1-day volume confirmation
Trades breakouts above upper Keltner Channel (EMA20 + 2*ATR) when weekly trend is up
and volume exceeds 1-day average. Exits on lower channel touch.
Designed for trending markets with volatility filtering to avoid whipsaws.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Keltner calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR(10) on 12h
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate EMA(20) on 12h
    ema = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: EMA(20) ± 2*ATR(10)
    upper = ema + 2 * atr
    lower = ema - 2 * atr
    
    # Align Keltner channels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate volume MA(20) on 1d
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need all indicators
    start_idx = max(10, 20, 50)  # ATR, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        weekly_trend = ema_50_1w_aligned[i]
        
        # Current Keltner levels
        upper_now = upper_aligned[i]
        lower_now = lower_aligned[i]
        
        # Volume filter: volume > 1.5x 1-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: breakout above upper Keltner with volume and weekly uptrend
        if position == 0:
            if price_now > upper_now and vol_filter and price_now > weekly_trend:
                signals[i] = size
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches lower Keltner or weekly trend turns down
            if price_now < lower_now or price_now < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
    
    return signals

name = "12h_KeltnerBreakout_Volume_1wTrend"
timeframe = "12h"
leverage = 1.0