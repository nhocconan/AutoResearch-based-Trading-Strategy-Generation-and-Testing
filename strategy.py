#!/usr/bin/env python3
# Hypothesis: 6h Donchian channel breakout with 12h trend filter and volume confirmation.
# Uses Donchian(20) to identify breakouts, filters by 12h EMA(50) trend direction,
# and requires volume > 1.8x 20-period average to confirm. Designed for 6h timeframe
# with 50-150 total trades over 4 years to minimize fee drag. Works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian Channel (20)
    donch_period = 20
    upper_donch = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_donch = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Volume filter: volume > 1.8x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_period, 50, 20)  # Wait for Donchian, EMA, and volume
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_donch[i]) or 
            np.isnan(lower_donch[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA(50)
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry conditions: breakout from Donchian in trend direction with volume
        long_breakout = close[i] > upper_donch[i]
        short_breakout = close[i] < lower_donch[i]
        
        long_entry = long_breakout and uptrend and volume_confirm[i]
        short_entry = short_breakout and downtrend and volume_confirm[i]
        
        # Exit conditions: opposite Donchian breakout or loss of trend
        long_exit = (close[i] < lower_donch[i]) or (not uptrend)
        short_exit = (close[i] > upper_donch[i]) or (not downtrend)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_DonchianBreakout_12hEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0