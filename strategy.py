#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with daily volume confirmation and weekly trend filter.
Enters long on breakout above 20-period high with volume > 1.5x daily average and weekly uptrend.
Enters short on breakdown below 20-period low with volume > 1.5x daily average and weekly downtrend.
Uses weekly timeframe for trend structure to reduce noise and avoid false signals.
Designed to work in both bull and bear markets by following the weekly trend while using
Donchian breakouts for entry timing and volume for conviction. Target: 20-40 trades/year per
symbol to minimize fee drag and avoid overtrading.
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
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channel (20-period)
    # Upper band: highest high of last 20 periods
    high_series = pd.Series(high)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    low_series = pd.Series(low)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian components, volume MA, and weekly EMA
    start_idx = max(20, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = ema_20_1w_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Breakout conditions
        breakout_up = price_now > donchian_upper[i-1]  # Break above previous period's upper band
        breakdown_down = price_now < donchian_lower[i-1]  # Break below previous period's lower band
        
        # Entry conditions
        if position == 0:
            # Long: breakout above upper band with volume + weekly uptrend
            if breakout_up and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: breakdown below lower band with volume + weekly downtrend
            elif breakdown_down and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: breakdown below lower band or weekly trend turns down
            if breakdown_down or price_now < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: breakout above upper band or weekly trend turns up
            if breakout_up or price_now > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dVolume_1wTrend"
timeframe = "4h"
leverage = 1.0