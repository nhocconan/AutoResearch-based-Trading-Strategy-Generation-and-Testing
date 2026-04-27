#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_Trend_Volume
Hypothesis: Weekly Donchian(20) breakouts with daily trend filter (EMA50) and volume confirmation capture strong momentum moves. Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by filtering with trend. Target: 15-25 trades/year per symbol.
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
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly Donchian(20) - using weekly high/low
    high_weekly = pd.Series(df_weekly['high'].values)
    low_weekly = pd.Series(df_weekly['low'].values)
    donchian_high = high_weekly.rolling(window=20, min_periods=20).max().values
    donchian_low = low_weekly.rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_daily = pd.Series(df_daily['close'].values)
    ema50_daily = close_daily.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Volume spike detection (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period - need 20 for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume spike and uptrend
            if (close[i] > donchian_high_aligned[i] and volume_spike[i] and 
                close[i] > ema50_daily_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume spike and downtrend
            elif (close[i] < donchian_low_aligned[i] and volume_spike[i] and 
                  close[i] < ema50_daily_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below weekly Donchian low or trend fails
            if (close[i] < donchian_low_aligned[i] or 
                close[i] < ema50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above weekly Donchian high or trend fails
            if (close[i] > donchian_high_aligned[i] or 
                close[i] > ema50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0