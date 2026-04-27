# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Hypothesis: 1-day Donchian(20) breakout with volume confirmation and 1-week trend filter.
Trades only during high-volume breakouts in the direction of the weekly trend.
Designed to work in both bull and bear markets by using the weekly trend as filter.
Target: 5-15 trades/year per symbol (20-60 total over 4 years) to minimize fee drag.
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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Donchian channel and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian(20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(high_1d), np.nan)
    for i in range(20, len(high_1d)):
        upper[i] = np.max(high_1d[i-20:i])
        lower[i] = np.min(low_1d[i-20:i])
    donch_upper_1d = upper
    donch_lower_1d = lower
    donch_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_upper_1d)
    donch_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_lower_1d)
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian, volume MA, and weekly EMA
    start_idx = max(20, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_1d_aligned[i]) or np.isnan(donch_lower_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper = donch_upper_1d_aligned[i]
        lower = donch_lower_1d_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        weekly_trend = ema_50_1w_aligned[i]
        
        # Volume filter: volume > 2x daily average (strict to reduce trades)
        vol_filter = vol_now > 2.0 * vol_ma
        
        # Entry conditions: breakout with volume and weekly trend alignment
        if position == 0:
            # Long: break above upper band + volume + weekly uptrend
            if close[i] > upper and vol_filter and close[i] > weekly_trend:
                signals[i] = size
                position = 1
            # Short: break below lower band + volume + weekly downtrend
            elif close[i] < lower and vol_filter and close[i] < weekly_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below weekly EMA or Donchian lower band
            if close[i] < weekly_trend or close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above weekly EMA or Donchian upper band
            if close[i] > weekly_trend or close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_VolumeBreakout_WeeklyTrendFilter"
timeframe = "1d"
leverage = 1.0