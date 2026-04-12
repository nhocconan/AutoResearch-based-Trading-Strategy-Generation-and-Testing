#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Breakout_Volume_Regime
Hypothesis: 1h momentum breakouts aligned with daily Camarilla S4/R4 levels and 4h trend filter, with volume confirmation and session filter (08-20 UTC) to reduce noise. Designed for low trade frequency (15-37/year) by requiring strong breakouts, trend alignment, volume surge, and active session. Works in bull/bear via 4h trend filter and mean-reversion exit at daily pivot.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Breakout_Volume_Regime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Daily Camarilla levels (S4/R4 for stronger breakouts)
    r4_1d = close_1d + range_1d * 1.5
    s4_1d = close_1d - range_1d * 1.5
    
    # === 4H EMA(21) FOR TREND FILTER ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    if len(close_4h) >= 21:
        ema_21_4h = np.zeros_like(close_4h)
        ema_21_4h[0] = close_4h[0]
        alpha = 2.0 / (21 + 1)
        for i in range(1, len(close_4h)):
            ema_21_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_21_4h[i-1]
    else:
        ema_21_4h = np.full_like(close_4h, np.nan)
    
    # Align daily and 4h data to 1h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Volume average (24-period for 1h = 1 day) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 24:
            vol_sum -= volume[i-24]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema_21_4h_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten or hold flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: at least 2.0x average
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Trend filter: price above/below 4h EMA(21)
        price_above_ema = close[i] > ema_21_4h_aligned[i]
        price_below_ema = close[i] < ema_21_4h_aligned[i]
        
        # Breakout entries at daily S4/R4 with volume and trend filters
        long_setup = (close[i] > r4_1d_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < s4_1d_aligned[i]) and vol_confirm and price_below_ema
        
        # Exit when price returns to daily pivot (mean reversion)
        exit_long = close[i] < pivot_1d_aligned[i]
        exit_short = close[i] > pivot_1d_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals