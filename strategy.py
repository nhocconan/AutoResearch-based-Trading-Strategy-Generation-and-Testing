#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1 levels
    r1 = close_prev + 1.1 * (high_prev - low_prev) / 12
    s1 = close_prev - 1.1 * (high_prev - low_prev) / 12
    
    # Align daily levels to 1h timeframe (with 1-day delay for completed bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h trend filter: EMA50
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume filter: current volume > 1.5x 20-period average (1h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # 3 hours cooldown
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 4h trend direction
        trend_up = close > ema_50_4h_aligned[i]
        trend_down = close < ema_50_4h_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars and session_filter[i]:
            # Long: Break above R1 in uptrend with volume
            if (close[i] > r1_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.20
                position = 1
                bars_since_last_trade = 0
            # Short: Break below S1 in downtrend with volume
            elif (close[i] < s1_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.20
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price re-enters Camarilla body (between R1 and S1) or trend change
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price re-enters Camarilla body or trend change
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: Camarilla R1/S1 breakouts with 4h trend alignment capture institutional
# breakouts in both bull and bear markets. The 1h timeframe provides precise entry
# timing while using 4h for trend direction and daily for support/resistance levels.
# Volume confirmation ensures genuine institutional participation. Session filter
# (08-20 UTC) reduces noise. Conservative sizing (0.20) limits drawdown. Target:
# 60-150 total trades over 4 years to minimize fee drag.