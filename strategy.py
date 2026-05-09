#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyPivotBreakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Get 12h data for Donchian and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot from previous week's OHLC
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = np.nan
    prev_weekly_low[0] = np.nan
    prev_weekly_close[0] = np.nan
    
    prev_weekly_range = prev_weekly_high - prev_weekly_low
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    weekly_r4 = weekly_pivot + 1.1 * prev_weekly_range * 1.1  # R4 = pivot + 1.1*range*1.1
    weekly_s4 = weekly_pivot - 1.1 * prev_weekly_range * 1.1  # S4 = pivot - 1.1*range*1.1
    
    # Align weekly pivot levels to 6h
    weekly_r4_6h = align_htf_to_ltf(prices, df_weekly, weekly_r4)
    weekly_s4_6h = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    
    # Calculate 12h Donchian channel (20-period)
    donchian_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_6h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_6h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike detection (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r4_6h[i]) or np.isnan(weekly_s4_6h[i]) or 
            np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(ema50_12h_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above weekly R4 AND above 12h Donchian high with uptrend and volume spike
            if (close[i] > weekly_r4_6h[i] and close[i] > donchian_high_6h[i] and 
                close[i] > ema50_12h_6h[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S4 AND below 12h Donchian low with downtrend and volume spike
            elif (close[i] < weekly_s4_6h[i] and close[i] < donchian_low_6h[i] and 
                  close[i] < ema50_12h_6h[i] and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below weekly S4 OR below 12h Donchian low OR trend turns down
            if (close[i] < weekly_s4_6h[i] or close[i] < donchian_low_6h[i] or 
                close[i] < ema50_12h_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above weekly R4 OR above 12h Donchian high OR trend turns up
            if (close[i] > weekly_r4_6h[i] or close[i] > donchian_high_6h[i] or 
                close[i] > ema50_12h_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals