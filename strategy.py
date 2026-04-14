#!/usr/bin/env python3
"""
Hypothesis: 1-day strategy using 1-week Bollinger Band breakout with volume confirmation and 4-hour trend filter.
Long when price breaks above 1-week upper BB + 4h EMA21 > EMA50 + volume surge.
Short when price breaks below 1-week lower BB + 4h EMA21 < EMA50 + volume surge.
Exit when price returns to 1-week middle BB or trend reverses.
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
    
    # Load 1-week data once for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1-week Bollinger Bands (20, 2)
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    middle_bb = sma_20
    
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
        # 1-week index (7 days per week: 7*24 = 168 1h bars, 168/24 = 7 1d bars per week)
        idx_1w = i // 7
        if idx_1w < 20:  # need enough for BB
            continue
        
        # Get previous 1-week BB values to avoid look-ahead
        upper_prev = upper_bb[idx_1w - 1] if idx_1w - 1 < len(upper_bb) else upper_bb[-1]
        lower_prev = lower_bb[idx_1w - 1] if idx_1w - 1 < len(lower_bb) else lower_bb[-1]
        middle_prev = middle_bb[idx_1w - 1] if idx_1w - 1 < len(middle_bb) else middle_bb[-1]
        if np.isnan(upper_prev) or np.isnan(lower_prev) or np.isnan(middle_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        upper_arr = np.full(len(df_1w), upper_prev)
        lower_arr = np.full(len(df_1w), lower_prev)
        middle_arr = np.full(len(df_1w), middle_prev)
        upper_1d = align_htf_to_ltf(prices, df_1w, upper_arr)[i]
        lower_1d = align_htf_to_ltf(prices, df_1w, lower_arr)[i]
        middle_1d = align_htf_to_ltf(prices, df_1w, middle_arr)[i]
        
        # 4-hour index (6 bars per day: 24/4 = 6)
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
            # Long: price breaks above 1w upper BB + 4h EMA21 > EMA50 + volume surge
            if (close[i] > upper_1d and 
                ema21_1d > ema50_1d and 
                volume[i] > vol_ma[i] * 2.0):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 1w lower BB + 4h EMA21 < EMA50 + volume surge
            elif (close[i] < lower_1d and 
                  ema21_1d < ema50_1d and 
                  volume[i] > vol_ma[i] * 2.0):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price returns to 1w middle BB or trend reverses
            if close[i] < middle_1d or ema21_1d < ema50_1d:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price returns to 1w middle BB or trend reverses
            if close[i] > middle_1d or ema21_1d > ema50_1d:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_1w_BB_4h_EMA_Volume"
timeframe = "1d"
leverage = 1.0