#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 12h EMA50 trend filter and volume confirmation.
# Williams %R < -80 (oversold) for long, > -20 (overbought) for short.
# Requires alignment with 12h EMA50 trend and volume > 1.5x 24-period average.
# Designed for low trade frequency (15-30/year) to avoid fee drag in 4h timeframe.
# Works in both bull/bear markets by requiring trend alignment and extreme reversals.
name = "4h_WilliamsR_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Volume filter: > 1.5x 24-period average (6 days of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), price above 12h EMA50 (uptrend), volume confirmation
            if (williams_r[i] < -80 and 
                close[i] > ema_50_4h[i] and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), price below 12h EMA50 (downtrend), volume confirmation
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_4h[i] and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -20 (overbought) or trend reversal
            if williams_r[i] > -20 or close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -80 (oversold) or trend reversal
            if williams_r[i] < -80 or close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals