#!/usr/bin/env python3
"""
Hypothesis: 4-hour strategy using 1-day Bollinger Band breakout with volume confirmation and 1-week trend filter.
Long when price closes above upper Bollinger Band (20,2) with volume > 1.5x 20-period average and 1-week close > 1-week EMA50.
Short when price closes below lower Bollinger Band (20,2) with volume > 1.5x 20-period average and 1-week close < 1-week EMA50.
Exit when price returns to middle Bollinger Band (20) or 1-week trend reverses.
Designed for low turnover: ~20-30 trades/year per symbol to minimize fee drag.
Uses proven Bollinger breakout pattern with multi-timeframe trend filter to work in both bull and bear markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data once for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    middle = sma
    
    # Load 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # 1-day index (16 bars per day for 4h timeframe)
        idx_1d = i // 16
        if idx_1d < bb_period:
            continue
        
        # Use previous 1d values to avoid look-ahead (previous completed bar)
        prev_idx = idx_1d - 1
        if prev_idx < 0:
            continue
            
        # Get Bollinger Bands from previous 1d bar
        upper_prev = upper[prev_idx] if prev_idx < len(upper) else upper[-1]
        lower_prev = lower[prev_idx] if prev_idx < len(lower) else lower[-1]
        middle_prev = middle[prev_idx] if prev_idx < len(middle) else middle[-1]
        
        # 1-week index (112 bars per week for 4h timeframe)
        idx_1w = i // 112
        if idx_1w < 50:
            continue
        
        # Use previous 1w values to avoid look-ahead
        prev_idx_1w = idx_1w - 1
        if prev_idx_1w < 0:
            continue
            
        # Get EMA50 from previous 1w bar
        ema50_prev = ema50[prev_idx_1w] if prev_idx_1w < len(ema50) else ema50[-1]
        
        if np.isnan(upper_prev) or np.isnan(lower_prev) or np.isnan(middle_prev) or np.isnan(ema50_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        upper_arr = np.full(len(df_1d), upper_prev)
        lower_arr = np.full(len(df_1d), lower_prev)
        middle_arr = np.full(len(df_1d), middle_prev)
        ema50_arr = np.full(len(df_1w), ema50_prev)
        
        upper_4h = align_htf_to_ltf(prices, df_1d, upper_arr)[i]
        lower_4h = align_htf_to_ltf(prices, df_1d, lower_arr)[i]
        middle_4h = align_htf_to_ltf(prices, df_1d, middle_arr)[i]
        ema50_4h = align_htf_to_ltf(prices, df_1w, ema50_arr)[i]
        
        if position == 0:
            # Long: price closes above upper BB + volume surge + 1w uptrend
            if (close[i] > upper_4h and 
                volume[i] > vol_ma[i] * 1.5 and
                close_1d[-1] > ema50_prev if len(close_1d) > 0 else False):  # Use latest available 1d close
                position = 1
                signals[i] = position_size
            # Short: price closes below lower BB + volume surge + 1w downtrend
            elif (close[i] < lower_4h and 
                  volume[i] > vol_ma[i] * 1.5 and
                  close_1d[-1] < ema50_prev if len(close_1d) > 0 else False):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price returns to middle BB or 1w trend turns down
            if close[i] < middle_4h or (len(close_1d) > 0 and close_1d[-1] < ema50_prev):
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price returns to middle BB or 1w trend turns up
            if close[i] > middle_4h or (len(close_1d) > 0 and close_1d[-1] > ema50_prev):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Bollinger_Breakout_1wTrend"
timeframe = "4h"
leverage = 1.0