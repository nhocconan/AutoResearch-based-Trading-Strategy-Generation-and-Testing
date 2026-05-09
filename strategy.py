#!/usr/bin/env python3
# 1D_MultiTF_Confluence
# Hypothesis: Daily timeframe strategy using weekly EMA trend filter, daily Donchian breakout with volume confirmation, and volatility regime filter. 
# Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation to avoid false breakouts.
# Targets 30-100 trades over 4 years (7-25/year) with low turnover to minimize fee drag.

name = "1D_MultiTF_Confluence"
timeframe = "1d"
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
    
    # Get weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily (they are already daily, but align for safety)
    highest_high_1d = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_1d = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    # Volatility filter: ATR(14) < ATR(50) to avoid high volatility periods
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr14 < atr50  # Only trade when short-term ATR is below long-term ATR
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1d[i]) or np.isnan(highest_high_1d[i]) or np.isnan(lowest_low_1d[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > highest_high_1d[i-1]  # Break above Donchian high
        short_breakout = close[i] < lowest_low_1d[i-1]   # Break below Donchian low
        
        trend_up = close[i] > ema_20_1d[i]
        trend_down = close[i] < ema_20_1d[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume + volatility filter
            if long_breakout and trend_up and volume_filter[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume + volatility filter
            elif short_breakout and trend_down and volume_filter[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout below Donchian low or trend reversal
            if close[i] < lowest_low_1d[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout above Donchian high or trend reversal
            if close[i] > highest_high_1d[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals