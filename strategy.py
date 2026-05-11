#!/usr/bin/env python3
"""
12h_Donchian_Breakout_1dTrend
Hypothesis: Donchian(20) breakout on 12h with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above 20-period high + 1d EMA50 uptrend + volume > 20-period average.
Short when price breaks below 20-period low + 1d EMA50 downtrend + volume > 20-period average.
Exit when price breaks opposite Donchian band or trend reverses.
Targets 15-30 trades/year (60-120 over 4 years) to minimize fee drift.
Works in bull by riding breakouts, in bear by catching breakdowns with trend filter.
"""

name = "12h_Donchian_Breakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Donchian Channels on 12h (20-period) ---
    period = 20
    # Highest high over last 20 periods
    donchian_high = pd.Series(high_12h).rolling(window=period, min_periods=period).max().values
    # Lowest low over last 20 periods
    donchian_low = pd.Series(low_12h).rolling(window=period, min_periods=period).min().values
    
    # --- Volume Confirmation: 12h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_12h[i] > ema50_1d_aligned[i]
        trend_down = close_12h[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_12h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for breakout entries in direction of 1d trend with volume
            if close_12h[i] > donchian_high[i] and trend_up and vol_ok:
                # Long breakout: price above 20-period high + 1d uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_12h[i] < donchian_low[i] and trend_down and vol_ok:
                # Short breakdown: price below 20-period low + 1d downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price breaks below Donchian low OR trend turns down
                if close_12h[i] < donchian_low[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above Donchian high OR trend turns up
                if close_12h[i] > donchian_high[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals