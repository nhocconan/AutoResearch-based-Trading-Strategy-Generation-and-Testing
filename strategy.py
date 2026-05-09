#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and trend
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 20-period weekly Donchian channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily
    upper_20_daily = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_daily = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_daily = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike detection (20-period for daily)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_daily[i]) or np.isnan(lower_20_daily[i]) or 
            np.isnan(ema34_1w_daily[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above weekly Donchian upper with uptrend and volume spike
            if close[i] > upper_20_daily[i] and close[i] > ema34_1w_daily[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly Donchian lower with downtrend and volume spike
            elif close[i] < lower_20_daily[i] and close[i] < ema34_1w_daily[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below weekly Donchian lower OR trend turns down
            if close[i] < lower_20_daily[i] or close[i] < ema34_1w_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above weekly Donchian upper OR trend turns up
            if close[i] > upper_20_daily[i] or close[i] > ema34_1w_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals