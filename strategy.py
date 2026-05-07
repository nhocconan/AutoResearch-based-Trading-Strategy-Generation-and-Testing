#!/usr/bin/env python3
name = "1d_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    # Calculate Weekly Pivot levels from previous week
    high_prev_week = np.roll(high, 7)
    low_prev_week = np.roll(low, 7)
    close_prev_week = np.roll(close, 7)
    # Set first 7 values to nan (no previous week data)
    high_prev_week[:7] = np.nan
    low_prev_week[:7] = np.nan
    close_prev_week[:7] = np.nan
    
    pivot_point = (high_prev_week + low_prev_week + close_prev_week) / 3.0
    weekly_range = high_prev_week - low_prev_week
    
    # Weekly Resistance 1 and Support 1
    weekly_r1 = 2 * pivot_point - low_prev_week
    weekly_s1 = 2 * pivot_point - high_prev_week
    
    # Volume spike: current volume > 2.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # 2 days to prevent overtrading
    
    start_idx = max(7, 20)  # Ensure enough data for weekly pivot and volume
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_r1[i]) or 
            np.isnan(weekly_s1[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Weekly R1 with volume spike in 1w uptrend
            if (close[i] > weekly_r1[i] and 
                trending_up and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Weekly S1 with volume spike in 1w downtrend
            elif (close[i] < weekly_s1[i] and 
                  trending_down and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Weekly S1 or 1w trend changes to down
            if close[i] < weekly_s1[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above Weekly R1 or 1w trend changes to up
            if close[i] > weekly_r1[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 1d timeframe, price breaking above/below Weekly R1/S1 levels with volume spike confirmation and 1w EMA50 trend filter captures institutional breakout momentum. Weekly pivot levels represent key weekly support/resistance derived from previous week's price action, reducing false breakouts. 1w trend filter ensures alignment with higher timeframe momentum. Volume spike filter (2.5x 20-period average) confirms institutional participation. Cooldown period prevents overtrading. Target: 30-80 total trades over 4 years (7-20/year) to minimize fee drag. Works in bull markets (breakouts above Weekly R1 in 1w uptrend) and bear markets (breakdowns below Weekly S1 in 1w downtrend). Uses discrete position sizing (0.25) to balance risk and reward while reducing fee churn. This strategy focuses on proven weekly pivot breakout with volume/trend confluence, which has shown strong performance in similar timeframes.