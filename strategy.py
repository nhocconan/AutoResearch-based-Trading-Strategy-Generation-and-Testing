#!/usr/bin/env python3
"""
12h_weekly_donchian_20_1d_trend
Hypothesis: On 12-hour timeframe, use weekly Donchian channel breakouts with daily trend filter.
Long when price breaks above weekly Donchian high with daily EMA(20) trending up.
Short when price breaks below weekly Donchian low with daily EMA(20) trending down.
Exit when price returns to the Donchian midpoint.
Designed for 15-25 trades/year to minimize fee decay while capturing major trends.
Weekly structure filters noise, daily trend avoids counter-trend trades, works in bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_weekly_donchian_20_1d_trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for Donchian channel
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian Channel (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    donchian_high = weekly_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = weekly_low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_weekly, donchian_mid)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(20) for trend filter
    daily_close = df_daily['close'].values
    daily_close_series = pd.Series(daily_close)
    ema_20_daily = daily_close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_20_daily)
    
    # Determine daily trend direction (using EMA slope)
    daily_trend_up = np.zeros(len(ema_20_daily_aligned), dtype=bool)
    daily_trend_down = np.zeros(len(ema_20_daily_aligned), dtype=bool)
    for i in range(1, len(ema_20_daily_aligned)):
        if not np.isnan(ema_20_daily_aligned[i]) and not np.isnan(ema_20_daily_aligned[i-1]):
            daily_trend_up[i] = ema_20_daily_aligned[i] > ema_20_daily_aligned[i-1]
            daily_trend_down[i] = ema_20_daily_aligned[i] < ema_20_daily_aligned[i-1]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_20_daily_aligned[i])):
            signals[i] = 0.0
            continue
            
        if position == 1:  # Long position
            # Exit: price returns to weekly Donchian midpoint
            if close[i] <= donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly Donchian midpoint
            if close[i] >= donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with daily trend alignment
            # Long: price breaks above weekly Donchian high with daily uptrend
            if (close[i] > donchian_high_aligned[i] and close[i-1] <= donchian_high_aligned[i-1] and 
                daily_trend_up[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly Donchian low with daily downtrend
            elif (close[i] < donchian_low_aligned[i] and close[i-1] >= donchian_low_aligned[i-1] and 
                  daily_trend_down[i]):
                position = -1
                signals[i] = -0.25
    
    return signals