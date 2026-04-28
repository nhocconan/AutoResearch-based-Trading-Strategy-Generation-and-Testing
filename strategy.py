#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Weekly_Filter
Hypothesis: Uses KAMA (adaptive moving average) on daily timeframe with weekly trend filter and volume confirmation to capture major trend moves while avoiding whipsaws. Designed for very low trade frequency (7-25/year) to minimize fee decay. KAMA adapts to market efficiency, reducing lag in trends and increasing noise filtering in ranges. Weekly filter ensures alignment with higher-timeframe momentum, improving win rate in both bull and bear markets.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate KAMA on daily close
    # KAMA parameters: ER fast=2, slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    # Efficiency Ratio
    price_change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility_sum > 0, price_change / volatility_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(kama[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]
        weekly_downtrend = ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]
        
        # KAMA direction
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry conditions
        long_entry = price_above_kama and weekly_uptrend and vol_confirm
        short_entry = price_below_kama and weekly_downtrend and vol_confirm
        
        # Exit conditions: reverse signal
        long_exit = price_below_kama or not weekly_uptrend
        short_exit = price_above_kama or not weekly_downtrend
        
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
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Trend_With_Weekly_Filter"
timeframe = "1d"
leverage = 1.0