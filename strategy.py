#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal with Daily Trend Filter and Volume Spike.
Long when price touches S1 level in bullish daily trend + volume spike.
Short when price touches R1 level in bearish daily trend + volume spike.
Exit when price moves back toward pivot (PP) level.
Designed for low frequency (12-37 trades/year) to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: PP = (H+L+C)/3, Range = H-L
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pp + (range_1d * 1.0 / 6.0)
    s1 = pp - (range_1d * 1.0 / 6.0)
    
    # Align to 12h timeframe (use previous day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: volume > 2x average (from 12h volume MA20)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily data (1 day) and volume MA (20 periods)
    start_idx = max(1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        pp_now = pp_aligned[i]
        r1_now = r1_aligned[i]
        s1_now = s1_aligned[i]
        daily_trend = ema_34_aligned[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price at S1 + daily trend up + volume spike
            if price_now <= s1_now and price_now > daily_trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price at R1 + daily trend down + volume spike
            elif price_now >= r1_now and price_now < daily_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price moves back toward pivot
            if price_now >= pp_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price moves back toward pivot
            if price_now <= pp_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_S1R1_DailyTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0