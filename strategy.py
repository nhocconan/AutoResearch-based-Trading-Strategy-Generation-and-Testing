#!/usr/bin/env python3
name = "12h_Camarilla_W1_Pivot_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Camarilla pivot levels from previous week
    camarilla_h1_w = np.full_like(close_1w, np.nan)
    camarilla_l1_w = np.full_like(close_1w, np.nan)
    
    for i in range(len(close_1w)):
        if i >= 1:
            prev_high = high_1w[i-1]
            prev_low = low_1w[i-1]
            prev_close = close_1w[i-1]
            range_ = prev_high - prev_low
            camarilla_h1_w[i] = prev_close + 1.1 * range_ / 12
            camarilla_l1_w[i] = prev_close - 1.1 * range_ / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h1_w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h1_w)
    camarilla_l1_w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l1_w)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    if len(close_1d) >= 50:
        ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema50_1d = np.full_like(close_1d, np.nan)
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h volume EMA20
    if len(volume_4h) >= 20:
        vol_ema20_4h = pd.Series(volume_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        vol_ema20_4h = np.full_like(volume_4h, np.nan)
    
    # Align 4h volume EMA20 to 12h timeframe
    vol_ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ema20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(1, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h1_w_aligned[i]) or np.isnan(camarilla_l1_w_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ema20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        # Downtrend: price below 1d EMA50
        downtrend = close[i] < ema50_1d_aligned[i]
        # Volume surge: current volume > 1.5x 4h volume EMA20
        volume_surge = volume[i] > vol_ema20_4h_aligned[i] * 1.5
        
        if position == 0:
            # Enter long: Uptrend + price touches/breaks above Camarilla H1 (weekly) + volume surge
            if uptrend and close[i] >= camarilla_h1_w_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price touches/breaks below Camarilla L1 (weekly) + volume surge
            elif downtrend and close[i] <= camarilla_l1_w_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price touches/breaks below Camarilla L1 (weekly)
            if not uptrend or close[i] <= camarilla_l1_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price touches/breaks above Camarilla H1 (weekly)
            if not downtrend or close[i] >= camarilla_h1_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals