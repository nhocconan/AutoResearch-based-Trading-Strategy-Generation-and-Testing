#!/usr/bin/env python3
"""
12h_1w_Donchian_Breakout_Trend_v1
Hypothesis: 12h Donchian(20) breakouts with 1w EMA trend filter and volume confirmation. Designed for low trade frequency (12-37/year) to avoid drag. Works in bull/bear via trend filter and ATR-based stop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Donchian_Breakout_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(21) for trend filter
    ema_21_1w = np.zeros_like(close_1w)
    ema_21_1w[0] = close_1w[0]
    alpha = 2.0 / (21 + 1)
    for i in range(1, len(close_1w)):
        ema_21_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_21_1w[i-1]
    
    # Align weekly EMA to 12h timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # 12h Donchian(20) channels
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i+1])
        lower[i] = np.min(low[i-20:i+1])
    
    # Volume average (10-period for 12h = 5 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 10:
            vol_sum -= volume[i-10]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if indicators not available
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 2.0x average
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Trend filter: price above/below weekly EMA(21)
        price_above_ema = close[i] > ema_21_1w_aligned[i]
        price_below_ema = close[i] < ema_21_1w_aligned[i]
        
        # Breakout entries at Donchian levels with volume and trend filters
        long_setup = (close[i] > upper[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < lower[i]) and vol_confirm and price_below_ema
        
        # Exit when price returns to opposite Donchian level (mean reversion)
        exit_long = close[i] < lower[i]
        exit_short = close[i] > upper[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals