#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
Trades only during high-volume breakouts in the direction of the daily trend.
Designed to work in both bull and bear markets by using the daily trend as filter.
Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(20) for trend
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 4-hour Donchian channels (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate 4-hour volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian, volume MA, and daily EMA
    start_idx = max(donchian_period, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper = highest_high[i]
        lower = lowest_low[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20[i]
        trend_1d = ema_20_1d_aligned[i]
        
        # Volume filter: volume > 1.8x 4h average (moderate to balance trades)
        vol_filter = vol_now > 1.8 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and daily trend alignment
        if position == 0:
            # Long: break above upper band + volume + daily uptrend
            if close[i] > upper and vol_filter and close[i] > trend_1d:
                signals[i] = size
                position = 1
            # Short: break below lower band + volume + daily downtrend
            elif close[i] < lower and vol_filter and close[i] < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below daily EMA or midpoint of Donchian
            midpoint = (upper + lower) / 2
            if close[i] < trend_1d or close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above daily EMA or midpoint of Donchian
            midpoint = (upper + lower) / 2
            if close[i] > trend_1d or close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DonchianBreakout_Volume_DailyTrendFilter"
timeframe = "4h"
leverage = 1.0