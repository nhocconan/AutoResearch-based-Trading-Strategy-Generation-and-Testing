#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Monthly Trend Filter and Volume Spike
Hypothesis: Weekly Donchian breakouts capture long-term trends. Using monthly EMA20 as trend filter ensures we only trade in the direction of the higher timeframe trend. Volume spikes confirm institutional participation. This strategy targets 10-25 trades per year to minimize fee drag and works in both bull and bear markets by following the trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_monthly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian Channel (20-period)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_high_roll = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_low_roll = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_roll)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_roll)
    
    # Monthly EMA20 Trend Filter
    df_monthly = get_htf_data(prices, '1M')
    monthly_close = df_monthly['close'].values
    monthly_ema20 = pd.Series(monthly_close).ewm(span=20, adjust=False).mean().values
    monthly_ema20_aligned = align_htf_to_ltf(prices, df_monthly, monthly_ema20)
    
    # Volume Spike Detector (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(monthly_ema20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly Donchian low
            if close[i] < weekly_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly Donchian high
            if close[i] > weekly_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above weekly Donchian high + price above monthly EMA20 + volume spike
            if (close[i] > weekly_high_aligned[i] and 
                close[i] > monthly_ema20_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: breakout below weekly Donchian low + price below monthly EMA20 + volume spike
            elif (close[i] < weekly_low_aligned[i] and 
                  close[i] < monthly_ema20_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals