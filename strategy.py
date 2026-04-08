#!/usr/bin/env python3
# 1d_weekly_ema_trend_volume_v2
# Hypothesis: Use weekly EMA (10) for trend direction on 1d timeframe. Enter long when price > weekly EMA10 and weekly EMA10 rising, short when price < weekly EMA10 and weekly EMA10 falling. Use volume confirmation (volume > 1.5x 20-day average) to filter noise. Exit when trend reverses. Weekly trend filter reduces whipsaw in ranging markets, capturing major trends in both bull and bear markets. Target: 20-50 total trades over 4 years (5-12.5/year) to minimize fee drag.

name = "1d_weekly_ema_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA(10) for trend direction
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=10, adjust=False, min_periods=10).mean().values
    weekly_ema_prev = np.roll(weekly_ema, 1)
    weekly_ema_prev[0] = np.nan
    weekly_ema_rising = weekly_ema > weekly_ema_prev
    weekly_ema_falling = weekly_ema < weekly_ema_prev
    
    # Align weekly EMA and trend to 1d timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    weekly_ema_rising_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema_rising.astype(float))
    weekly_ema_falling_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema_falling.astype(float))
    
    # Volume filter: volume > 1.5x 20-day average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(20, 10) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema_aligned[i]) or np.isnan(weekly_ema_rising_aligned[i]) or
            np.isnan(weekly_ema_falling_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: trend turns down
            if weekly_ema_falling_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns up
            if weekly_ema_rising_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long when price above rising weekly EMA
            if (close[i] > weekly_ema_aligned[i] and weekly_ema_rising_aligned[i] and volume_filter):
                position = 1
                signals[i] = 0.25
            # Short when price below falling weekly EMA
            elif (close[i] < weekly_ema_aligned[i] and weekly_ema_falling_aligned[i] and volume_filter):
                position = -1
                signals[i] = -0.25
    
    return signals