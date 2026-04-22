#!/usr/bin/env python3

"""
Hypothesis: Daily Close above 10-period SMA with weekly trend filter and volume confirmation.
Go long when daily close crosses above SMA10 and weekly EMA34 is rising; short when daily close crosses below SMA10 and weekly EMA34 is falling.
Requires volume confirmation (volume > 1.5x 20-period average) to avoid false breakouts.
Designed for low trade frequency (7-25 trades/year) by requiring multiple confirmations: trend alignment, price crossover, and volume spike.
Works in both bull and bear markets by following the weekly trend.
"""

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
    
    # 10-period SMA on daily
    close_s = pd.Series(close)
    sma10 = close_s.rolling(window=10, min_periods=10).mean().values
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    weekly_close = df_weekly['close'].values
    ema34_weekly = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        # Skip if data not ready
        if (np.isnan(sma10[i]) or np.isnan(ema34_weekly_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: close crosses above SMA10 + weekly uptrend + volume spike
            if close[i] > sma10[i] and close[i-1] <= sma10[i-1] and ema34_weekly_aligned[i] > ema34_weekly_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: close crosses below SMA10 + weekly downtrend + volume spike
            elif close[i] < sma10[i] and close[i-1] >= sma10[i-1] and ema34_weekly_aligned[i] < ema34_weekly_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses back through SMA10
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below SMA10
                if close[i] < sma10[i] and close[i-1] >= sma10[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above SMA10
                if close[i] > sma10[i] and close[i-1] <= sma10[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_SMA10_Cross_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0