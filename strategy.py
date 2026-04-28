#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_Daily_Trend_And_Volume_Filter
Hypothesis: KAMA adapts to market conditions, capturing trends while reducing whipsaw in ranging markets.
Combining with daily trend filter (price > daily EMA34) and volume confirmation creates robust trend-following
that works in both bull and bear markets. Target: 15-25 trades/year.
"""

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate KAMA (12h timeframe)
    # Efficiency Ratio = |change| / volatility
    change = np.abs(np.diff(close, k=10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    er = np.zeros_like(change)
    mask = vol != 0
    er[mask] = change[mask] / vol[mask]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        price_above_daily_ema = close[i] > ema_34_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions
        long_entry = price_above_kama and price_above_daily_ema and volume_filter[i]
        short_entry = price_below_kama and price_below_daily_ema and volume_filter[i]
        
        # Exit conditions (opposite signal)
        long_exit = price_below_kama and price_below_daily_ema
        short_exit = price_above_kama and price_above_daily_ema
        
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

name = "12h_KAMA_Trend_With_Daily_Trend_And_Volume_Filter"
timeframe = "12h"
leverage = 1.0