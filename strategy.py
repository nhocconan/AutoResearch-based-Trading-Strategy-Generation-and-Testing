#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) Breakout with Weekly Trend Filter and Volume Spike
# Uses daily Donchian channel breakouts for entry, weekly EMA20 for trend alignment,
# and volume spike for confirmation. Designed for 7-25 trades/year to avoid fee drag.
# Works in bull markets (breakouts with trend) and bear markets (fades from channel with trend).
name = "1d_Donchian20_WeeklyTrend_Volume"
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
    
    # Get weekly data for EMA trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema20_weekly = pd.Series(df_weekly['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Daily Donchian(20) channels
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1d[i]) or np.isnan(high_rolling[i]) or np.isnan(low_rolling[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above upper band with weekly uptrend and volume spike
            if close[i] > high_rolling[i] and close[i] > ema20_1d[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band with weekly downtrend and volume spike
            elif close[i] < low_rolling[i] and close[i] < ema20_1d[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below lower band OR weekly trend turns down
            if close[i] < low_rolling[i] or close[i] < ema20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above upper band OR weekly trend turns up
            if close[i] > high_rolling[i] or close[i] > ema20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals