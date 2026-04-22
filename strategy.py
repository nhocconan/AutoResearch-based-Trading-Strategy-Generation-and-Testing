#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams %R with 1-week trend filter and volume confirmation.
Long when Williams %R crosses above -20 (oversold) + weekly close > weekly SMA20 + volume > 1.5x average.
Short when Williams %R crosses below -80 (overbought) + weekly close < weekly SMA20 + volume > 1.5x average.
Exit when Williams %R crosses -50 (mean reversion) or weekly trend changes.
Designed for low trade frequency (~15-30/year) to minimize fee drift in both bull and bear markets.
"""

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
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly SMA20 for trend filter
    weekly_close = df_1w['close'].values
    weekly_sma20 = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    weekly_sma20_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma20)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    willr = willr.values  # Convert to numpy array
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(willr[i]) or np.isnan(weekly_sma20_aligned[i]) or 
            np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_close_val = None
        weekly_sma20_val = None
        if i < len(weekly_sma20_aligned):
            weekly_close_val = df_1w['close'].values[-1] if len(df_1w) > 0 else np.nan
            weekly_sma20_val = weekly_sma20_aligned[i]
        else:
            weekly_close_val = np.nan
            weekly_sma20_val = np.nan
            
        if np.isnan(weekly_close_val) or np.isnan(weekly_sma20_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        weekly_trend_up = weekly_close_val > weekly_sma20_val
        weekly_trend_down = weekly_close_val < weekly_sma20_val
        
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: Williams %R crosses above -20 + weekly uptrend + volume confirmation
            if (willr[i] > -20 and willr[i-1] <= -20 and 
                weekly_trend_up and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 + weekly downtrend + volume confirmation
            elif (willr[i] < -80 and willr[i-1] >= -80 and 
                  weekly_trend_down and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses below -50 or weekly trend changes to down
                if willr[i] < -50 or not weekly_trend_up:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses above -50 or weekly trend changes to up
                if willr[i] > -50 or not weekly_trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_WeeklyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0