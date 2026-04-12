#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_With_Volume_Confirmation
Hypothesis: Kaufman Adaptive Moving Average (KAMA) on daily chart identifies strong trending regimes,
while weekly trend filter ensures alignment with higher timeframe momentum. Volume spikes on
daily chart confirm institutional participation. Works in both bull (trend following) and bear
(avoiding false signals during chop) markets by requiring alignment between KAMA direction,
weekly trend, and volume confirmation. Target: 15-25 trades per year (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_Trend_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # === KAMA (Kaufman Adaptive Moving Average) on daily close ===
    # Fast EMA = 2/(2+1) = 0.6667, Slow EMA = 2/(30+1) = 0.0645
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate efficiency ratio
    change = np.abs(np.diff(daily_close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(daily_close)), axis=1)  # 10-period volatility
    er = np.where(volatility > 0, change / volatility, 0)
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(daily_close, np.nan)
    kama[9] = daily_close[9]  # Start after 10 periods
    for i in range(10, len(daily_close)):
        kama[i] = kama[i-1] + sc[i] * (daily_close[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (already aligned, but for consistency)
    kama_aligned = kama  # Already on 1d frequency
    
    # === Weekly trend filter (EMA 21) ===
    weekly_ema = np.full_like(weekly_close, np.nan)
    weekly_ema[20] = np.mean(weekly_close[:21])  # Simple average for first value
    for i in range(21, len(weekly_close)):
        weekly_ema[i] = weekly_ema[i-1] + (2/(21+1)) * (weekly_close[i] - weekly_ema[i-1])
    
    # Align weekly EMA to daily timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === Volume spike detection (2x 20-day average) ===
    vol_ma_20 = np.full_like(daily_volume, np.nan)
    for i in range(20, len(daily_volume)):
        vol_ma_20[i] = np.mean(daily_volume[i-20:i])
    vol_spike = daily_volume > (2 * vol_ma_20)
    vol_spike_aligned = vol_spike  # Already on 1d frequency
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after sufficient warmup
        # Skip if any data invalid
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(weekly_ema_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA direction: price above KAMA = uptrend, below = downtrend
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # Weekly trend filter: price above weekly EMA = bullish bias
        weekly_bullish = close[i] > weekly_ema_aligned[i]
        weekly_bearish = close[i] < weekly_ema_aligned[i]
        
        # Entry conditions
        long_entry = price_above_kama and weekly_bullish and vol_spike_aligned[i]
        short_entry = price_below_kama and weekly_bearish and vol_spike_aligned[i]
        
        # Exit conditions: reverse signal or price crosses KAMA in opposite direction
        long_exit = price_below_kama or not weekly_bullish
        short_exit = price_above_kama or not weekly_bearish
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals