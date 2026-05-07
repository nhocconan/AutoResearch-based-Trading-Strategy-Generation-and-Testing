#!/usr/bin/env python3

name = "12h_WeeklyCamarillaPivot_Squeeze_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R3 = pivot + (high_1d - low_1d) * 1.1 / 2
    S3 = pivot - (high_1d - low_1d) * 1.1 / 2
    
    # Align weekly Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Weekly trend filter using EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    close_1w = df_1w['close'].values
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    trend_up = close_1w_aligned > ema_50_1w_aligned
    trend_down = close_1w_aligned < ema_50_1w_aligned
    
    # Bollinger Bands squeeze filter (20, 2) on 12h data
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / ma_20
    # Squeeze when BB width is below 20-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # Prevent overtrading (approx 4 days for 12h)
    
    start_idx = max(20, 50)  # Warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ma_20[i]) or 
            np.isnan(std_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price breaks above S3 in weekly uptrend during BB squeeze
            if (close[i] > S3_aligned[i] and 
                trend_up[i] and 
                squeeze[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price breaks below R3 in weekly downtrend during BB squeeze
            elif (close[i] < R3_aligned[i] and 
                  trend_down[i] and 
                  squeeze[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price closes below S3 OR trend change OR BB expansion
            if (close[i] < S3_aligned[i]) or (not trend_up[i]) or (not squeeze[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above R3 OR trend change OR BB expansion
            if (close[i] > R3_aligned[i]) or (not trend_down[i]) or (not squeeze[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Camarilla pivot breakouts with BB squeeze filter.
# Long when price breaks above S3 level in weekly uptrend during Bollinger Band squeeze.
# Short when price breaks below R3 level in weekly downtrend during Bollinger Band squeeze.
# BB squeeze indicates low volatility primed for expansion breakout.
# Weekly EMA50 filter ensures trading with higher timeframe trend.
# Using 12h timeframe targets 15-35 trades/year to avoid fee drag.
# Works in both bull and bear markets by capturing volatility expansion in trend direction.