#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    # Hypothesis: Weekly Donchian breakout (20-week) with daily EMA34 trend and volume surge
    # Works in both bull and bear markets: weekly structure filters noise, 
    # breakouts from long-term channels capture major moves, volume confirms strength
    # EMA34 ensures alignment with daily trend direction
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-week lookback)
    highest_high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA34 trend filter
    daily_ema34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema34_aligned = align_htf_to_ltf(prices, df_1w, daily_ema34)
    
    # Volume filter (20-period surge on daily)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(300, n):
        # Skip if weekly data not ready
        if (np.isnan(highest_high_20w[i]) or np.isnan(lowest_low_20w[i]) or 
            np.isnan(daily_ema34_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly Donchian breakout above 20w high + volume surge + daily EMA34 uptrend
            if (prices['close'].values[i] > highest_high_20w[i] and 
                vol_surge[i] and 
                prices['close'].values[i] > daily_ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Weekly Donchian breakout below 20w low + volume surge + daily EMA34 downtrend
            elif (prices['close'].values[i] < lowest_low_20w[i] and 
                  vol_surge[i] and 
                  prices['close'].values[i] < daily_ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to weekly Donchian middle or opposite breakout
            if position == 1:
                mid_point = (highest_high_20w[i] + lowest_low_20w[i]) / 2
                if prices['close'].values[i] < mid_point:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                mid_point = (highest_high_20w[i] + lowest_low_20w[i]) / 2
                if prices['close'].values[i] > mid_point:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_20w_EMA34_Trend_VolumeSurge_v1"
timeframe = "1d"
leverage = 1.0