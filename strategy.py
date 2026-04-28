#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12-hour Camarilla R3/S3 level breakouts with 1-day EMA34 trend filter and volume spike work in both bull and bear markets. The trend filter reduces whipsaw, volume confirms institutional participation, and tight entry conditions limit trades to avoid fee drag. Target: 15-25 trades/year per symbol.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (HLC of previous day)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    #          S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    
    # We need previous day's HLC, so we'll calculate it manually
    # For each 12h bar, we need the previous day's high, low, close
    
    # Create arrays for previous day's HLC
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    prev_day_close = np.full(n, np.nan)
    
    # For simplicity, we'll use the daily data and align it
    # Each 12h bar corresponds to half a day, so we need to be careful
    
    # Instead, let's calculate Camarilla levels using rolling window on 12h data
    # But using the previous day's equivalent (24h = 2*12h bars)
    lookback = 2  # 2 * 12h = 24h = 1 day
    
    # Calculate rolling max/min/close for the lookback period
    roll_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    roll_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    roll_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).last().values
    
    # Calculate Camarilla levels
    H_L = roll_high - roll_low
    C = roll_close
    
    R3 = C + H_L * 1.1 / 4
    S3 = C - H_L * 1.1 / 4
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_long = close[i] > R3[i-1]  # Break above R3
        breakout_short = close[i] < S3[i-1]  # Break below S3
        
        # Trend filter from daily EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_long and volume_spike[i] and uptrend
        short_entry = breakout_short and volume_spike[i] and downtrend
        
        # Exit on opposite breakout (reverse position)
        long_exit = breakout_short and volume_spike[i]
        short_exit = breakout_long and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0