#!/usr/bin/env python3
"""
1d_1w_Donchian_Breakout_Volume_Filter
Hypothesis: On daily timeframe, breakouts of Donchian channel (20) with volume confirmation and 1-week trend filter.
- Long when: price breaks above Donchian(20) high, volume > 20-day average, and 1-week EMA50 uptrend
- Short when: price breaks below Donchian(20) low, volume > 20-day average, and 1-week EMA50 downtrend
- Exit when: price crosses Donchian midpoint OR trend reverses
Donchian captures breakouts, volume confirms participation, 1-week EMA filters for higher-timeframe trend.
Targets 10-25 trades/year (40-100 over 4 years) to minimize fee drag and work in both bull and bear markets.
"""

name = "1d_1w_Donchian_Breakout_Volume_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w Trend Filter: EMA50 ---
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- Donchian Channel (20) on 1d ---
    period = 20
    # Upper band: highest high of last 20 periods
    upper = np.full_like(high, np.nan)
    for i in range(period - 1, len(high)):
        upper[i] = np.max(high[i - period + 1:i + 1])
    # Lower band: lowest low of last 20 periods
    lower = np.full_like(low, np.nan)
    for i in range(period - 1, len(low)):
        lower[i] = np.min(low[i - period + 1:i + 1])
    # Middle band: average of upper and lower
    middle = (upper + lower) / 2.0
    
    # --- Volume Confirmation: 1d volume > 20-day average ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for Donchian (20) + volume MA (20) + 1w EMA (50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1w trend
        trend_up = close[i] > ema50_1w_aligned[i]
        trend_down = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for breakouts in direction of 1w trend with volume
            if close[i] > upper[i] and trend_up and vol_ok:
                # Breakout above Donchian high + 1w uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close[i] < lower[i] and trend_down and vol_ok:
                # Breakout below Donchian low + 1w downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below middle OR trend turns down
                if close[i] < middle[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above middle OR trend turns up
                if close[i] > middle[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals