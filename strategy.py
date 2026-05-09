#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 13/48 EMA crossover with weekly trend filter and volume confirmation
# Uses dual EMA for trend detection (13 fast, 48 slow) on daily timeframe
# Weekly EMA48 acts as trend filter to avoid counter-trend trades
# Volume confirmation requires current volume > 1.5x 20-day average
# Designed for 1d timeframe with target of 30-100 trades over 4 years (7-25/year)
# Works in bull/bear markets by requiring alignment with weekly trend
# EMA crossover provides timely entries while weekly filter reduces false signals
name = "1d_EMA13_48_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 48:
        return np.zeros(n)
    
    # Calculate weekly EMA48 trend filter
    ema_48_1w = pd.Series(df_1w['close'].values).ewm(span=48, adjust=False, min_periods=48).mean().values
    ema_48_1d = align_htf_to_ltf(prices, df_1w, ema_48_1w)
    
    # Calculate daily EMA13 and EMA48 for crossover
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_48 = pd.Series(close).ewm(span=48, adjust=False, min_periods=48).mean().values
    
    # Volume filter: current volume > 1.5x 20-day average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 48  # Need enough data for EMA calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or np.isnan(ema_48[i]) or np.isnan(ema_48_1d[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # EMA crossover signals
        ema_cross_up = ema_13[i] > ema_48[i]  # Fast EMA above slow EMA
        ema_cross_down = ema_13[i] < ema_48[i]  # Fast EMA below slow EMA
        
        weekly_uptrend = close[i] > ema_48_1d[i]
        weekly_downtrend = close[i] < ema_48_1d[i]
        
        if position == 0:
            # Long: bullish crossover + weekly uptrend + volume confirmation
            if ema_cross_up and weekly_uptrend and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish crossover + weekly downtrend + volume confirmation
            elif ema_cross_down and weekly_downtrend and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish crossover or weekly trend reversal
            if ema_cross_down or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish crossover or weekly trend reversal
            if ema_cross_up or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals