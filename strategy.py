#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian_Breakout_Trend_WeeklyVMA"
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian(20) - previous week's high/low
    weekly_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().shift(1).values
    weekly_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Weekly trend: EMA(34) of weekly close
    weekly_ema34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current week volume > 1.5 * 20-week average
    weekly_volume = df_1w['volume'].values
    vol_series = pd.Series(weekly_volume)
    weekly_vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    weekly_vol_filter = weekly_volume > (weekly_vol_ma * 1.5)
    
    # Align to daily
    weekly_high_d = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_d = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_ema34_d = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    weekly_vol_filter_d = align_htf_to_ltf(prices, df_1w, weekly_vol_filter)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # Need EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_d[i]) or np.isnan(weekly_low_d[i]) or
            np.isnan(weekly_ema34_d[i]) or np.isnan(weekly_vol_filter_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wh = weekly_high_d[i]
        wl = weekly_low_d[i]
        trend = weekly_ema34_d[i]
        vol_filter = weekly_vol_filter_d[i]
        
        if position == 0:
            # Long: break above weekly high with trend and volume
            if close[i] > wh and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly low with trend and volume
            elif close[i] < wl and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below weekly low
            if close[i] < wl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above weekly high
            if close[i] > wh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals