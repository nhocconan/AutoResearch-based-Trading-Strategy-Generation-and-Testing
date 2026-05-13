# 165128
#!/usr/bin/env python3
"""
12h_WM_Donchian_20_WeeklyTrend_VolumeFilter
Hypothesis: Weekly Donchian(20) breakout with weekly trend filter (price > weekly EMA20) and volume confirmation provides strong directional signals in 12h timeframe. Weekly timeframe reduces noise and false breakouts, while volume confirmation ensures institutional participation. Designed for low trade frequency (15-25/year) to minimize fee drag in 12-hour bars. Works in both bull and bear markets by following the weekly trend direction.
"""

name = "12h_WM_Donchian_20_WeeklyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period high/low)
    donchian_high = pd.Series(df_weekly['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_weekly['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Weekly trend filter: EMA(20) on close
    ema20_weekly = pd.Series(df_weekly['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Volume confirmation: current volume > 1.5x 24-period average (12 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        if position == 0:
            # LONG: Price breaks above weekly Donchian high, volume confirmation, price above weekly EMA20 (uptrend)
            if (close[i] > donchian_high_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema20_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low, volume confirmation, price below weekly EMA20 (downtrend)
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema20_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below weekly Donchian high (failed breakout) OR volume drops
            if (close[i] < donchian_high_aligned[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above weekly Donchian low (failed breakdown) OR volume drops
            if (close[i] > donchian_low_aligned[i] or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals