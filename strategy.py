#!/usr/bin/env python3
"""
Hypothesis: 1-day strategy using 1-week Donchian breakout with volume confirmation and 4-hour trend filter.
Long when price breaks above 1-week high + 4h EMA21 > EMA50 + volume surge.
Short when price breaks below 1-week low + 4h EMA21 < EMA50 + volume surge.
Exit when price returns to 1-week midline or trend reverses.
Designed for low turnover: ~10-20 trades/year per symbol to minimize fee drift.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data once for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-week Donchian channels (20)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Load 4-hour data once for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4-hour EMA21 and EMA50
    close_4h = df_4h['close'].values
    ema_21 = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # 1-week index (7 days per week)
        idx_1w = i // 7
        if idx_1w < 20:  # need enough for Donchian
            continue
        
        # Get previous 1-week Donchian values to avoid look-ahead
        high_prev = donch_high[idx_1w - 1] if idx_1w - 1 < len(donch_high) else donch_high[-1]
        low_prev = donch_low[idx_1w - 1] if idx_1w - 1 < len(donch_low) else donch_low[-1]
        mid_prev = donch_mid[idx_1w - 1] if idx_1w - 1 < len(donch_mid) else donch_mid[-1]
        if np.isnan(high_prev) or np.isnan(low_prev) or np.isnan(mid_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        high_arr = np.full(len(df_1w), high_prev)
        low_arr = np.full(len(df_1w), low_prev)
        mid_arr = np.full(len(df_1w), mid_prev)
        high_1d = align_htf_to_ltf(prices, df_1w, high_arr)[i]
        low_1d = align_htf_to_ltf(prices, df_1w, low_arr)[i]
        mid_1d = align_htf_to_ltf(prices, df_1w, mid_arr)[i]
        
        # 4-hour index (6 bars per day)
        idx_4h = i // 6
        if idx_4h < 50:  # need enough for EMA
            continue
        
        # Get previous 4h EMA values to avoid look-ahead
        ema21_prev = ema_21[idx_4h - 1] if idx_4h - 1 < len(ema_21) else ema_21[-1]
        ema50_prev = ema_50[idx_4h - 1] if idx_4h - 1 < len(ema_50) else ema_50[-1]
        if np.isnan(ema21_prev) or np.isnan(ema50_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        ema21_arr = np.full(len(df_4h), ema21_prev)
        ema50_arr = np.full(len(df_4h), ema50_prev)
        ema21_1d = align_htf_to_ltf(prices, df_4h, ema21_arr)[i]
        ema50_1d = align_htf_to_ltf(prices, df_4h, ema50_arr)[i]
        
        if position == 0:
            # Long: price breaks above 1w high + 4h EMA21 > EMA50 + volume surge
            if (close[i] > high_1d and 
                ema21_1d > ema50_1d and 
                volume[i] > vol_ma[i] * 2.0):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 1w low + 4h EMA21 < EMA50 + volume surge
            elif (close[i] < low_1d and 
                  ema21_1d < ema50_1d and 
                  volume[i] > vol_ma[i] * 2.0):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price returns to 1w mid or trend reverses
            if close[i] < mid_1d or ema21_1d < ema50_1d:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price returns to 1w mid or trend reverses
            if close[i] > mid_1d or ema21_1d > ema50_1d:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_1w_Donchian_4h_EMA_Volume"
timeframe = "1d"
leverage = 1.0