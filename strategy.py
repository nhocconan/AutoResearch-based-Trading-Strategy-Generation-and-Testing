#!/usr/bin/env python3
name = "1d_WeeklyTrend_WeeklyVolume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly EMA21 trend
    weekly_close = df_weekly['close'].values
    ema21_weekly = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_trend_up = ema21_weekly > np.roll(ema21_weekly, 1)  # rising
    weekly_trend_down = ema21_weekly < np.roll(ema21_weekly, 1)  # falling
    
    # Weekly volume filter
    weekly_volume = df_weekly['volume'].values
    vol_mean = np.mean(weekly_volume[-10:]) if len(weekly_volume) >= 10 else np.mean(weekly_volume)
    weekly_vol_high = weekly_volume > 1.5 * vol_mean
    
    # Align to daily
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_down)
    weekly_vol_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_vol_high)
    
    # Daily range filter: avoid extreme volatility
    daily_range = (high - low) / close
    range_mean = np.mean(daily_range[-20:]) if len(daily_range) >= 20 else np.mean(daily_range)
    range_filter = daily_range < 2.0 * range_mean  # not too volatile
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]) or
            np.isnan(weekly_vol_high_aligned[i]) or np.isnan(range_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend + high volume + not too volatile
            if (weekly_trend_up_aligned[i] and 
                weekly_vol_high_aligned[i] and 
                range_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + high volume + not too volatile
            elif (weekly_trend_down_aligned[i] and 
                  weekly_vol_high_aligned[i] and 
                  range_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down or volatility too high
            if (not weekly_trend_up_aligned[i] or not range_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up or volatility too high
            if (not weekly_trend_down_aligned[i] or not range_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals