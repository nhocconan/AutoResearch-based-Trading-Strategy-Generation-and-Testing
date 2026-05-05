#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d EMA50 trend filter + volume confirmation
# Williams %R: measures overbought/oversold levels (-20 to -80)
# Long when: Williams %R < -80 (oversold) AND price > EMA50 (1d uptrend) AND volume > 1.5x 20-period MA
# Short when: Williams %R > -20 (overbought) AND price < EMA50 (1d downtrend) AND volume > 1.5x 20-period MA
# Exit when: Williams %R reverses to opposite extreme (> -50 for long exit, < -50 for short exit)
# Uses Williams %R for momentum exhaustion, EMA for trend filter, volume for conviction
# Timeframe: 12h, HTF: 1d for EMA50. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_WilliamsR_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 12h
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when high == low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, np.nan)
    
    # Williams %R signals
    oversold = williams_r < -80   # Oversold condition
    overbought = williams_r > -20 # Overbought condition
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA50 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d
    close_1d = df_1d['close'].values
    if len(close_1d) >= 50:
        ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50 = np.full(len(df_1d), np.nan)
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: oversold + price above EMA50 + volume filter
            if (oversold[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: overbought + price below EMA50 + volume filter
            elif (overbought[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R reverses above -50 (leaving oversold territory)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R reverses below -50 (leaving overbought territory)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals