#!/usr/bin/env python3
"""
1d_1w_ThreeLegTrend_v1
Hypothesis: On daily timeframe, use weekly EMA(21) trend filter + daily Donchian(20) breakout + volume confirmation (2x average).
Trades only in direction of weekly trend to avoid counter-trend whipsaws. Designed for low frequency (10-25 trades/year) by requiring strong breakouts with trend and volume confirmation.
Works in bull/bear via weekly trend filter - only takes longs in weekly uptrend, shorts in weekly downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ThreeLegTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend filter
    if len(close_1w) >= 21:
        ema_21_1w = np.zeros_like(close_1w)
        ema_21_1w[0] = close_1w[0]
        alpha = 2.0 / (21 + 1)
        for i in range(1, len(close_1w)):
            ema_21_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_21_1w[i-1]
    else:
        ema_21_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly EMA to daily timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === DAILY INDICATORS ===
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume average (20-period)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 2.0x average
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Trend filter: price relative to weekly EMA(21)
        price_above_weekly_ema = close[i] > ema_21_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_21_1w_aligned[i]
        
        # Breakout entries with volume and trend filters
        long_setup = (close[i] > donchian_high[i]) and vol_confirm and price_above_weekly_ema
        short_setup = (close[i] < donchian_low[i]) and vol_confirm and price_below_weekly_ema
        
        # Exit when price crosses the 20-day EMA (trend change signal)
        ema_20 = np.zeros(n)
        if i >= 20:
            ema_20[i] = np.mean(close[i-20:i])
        exit_long = i >= 20 and close[i] < ema_20[i]
        exit_short = i >= 20 and close[i] > ema_20[i]
        
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