#!/usr/bin/env python3
"""
1d_1w_WeeklyDonchianBreakout_TrendVol_v1
Hypothesis: Weekly Donchian channel breakouts on daily timeframe with trend filter and volume confirmation.
In bull markets, price breaks above 20-week high with volume surge and weekly uptrend.
In bear markets, price breaks below 20-week low with volume surge and weekly downtrend.
Uses weekly timeframe for trend to reduce noise and focus on major trend changes.
"""

name = "1d_1w_WeeklyDonchianBreakout_TrendVol_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.8x 50-period average (daily)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Weekly Donchian channels (20 weeks)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly high and low for Donchian channels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate 20-period rolling high/low on weekly data
    weekly_high_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_low_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    weekly_high_20_aligned = align_htf_to_ltf(prices, df_1w, weekly_high_20)
    weekly_low_20_aligned = align_htf_to_ltf(prices, df_1w, weekly_low_20)
    
    # Weekly trend filter: EMA50 on weekly close
    weekly_ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(weekly_high_20_aligned[i]) or
            np.isnan(weekly_low_20_aligned[i]) or
            np.isnan(weekly_ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly Donchian high + volume spike + price above weekly EMA50
            if (close[i] > weekly_high_20_aligned[i] and 
                volume_spike[i] and 
                close[i] > weekly_ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low + volume spike + price below weekly EMA50
            elif (close[i] < weekly_low_20_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < weekly_ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters weekly Donchian channel OR closes below weekly EMA50
            if (close[i] < weekly_high_20_aligned[i] and close[i] > weekly_low_20_aligned[i]) or \
               close[i] < weekly_ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters weekly Donchian channel OR closes above weekly EMA50
            if (close[i] < weekly_high_20_aligned[i] and close[i] > weekly_low_20_aligned[i]) or \
               close[i] > weekly_ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals