#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w trend filter and Donchian(20) breakout with volume confirmation.
# Uses 1w EMA20 for trend filter and 1d Donchian(20) for breakout levels.
# Designed for low trade frequency (10-20/year) to avoid fee drag in 1d timeframe.
# Works in both bull/bear markets by requiring alignment with 1w trend and volatility breakout.
name = "1d_Donchian20_1wEMA20_VolumeBreakout"
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
    
    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1d Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: spike above 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1d[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price above Donchian(20) upper, 1w uptrend (price > EMA20), volume breakout
            if (close[i] > high_20[i] and 
                close[i] > ema_20_1d[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below Donchian(20) lower, 1w downtrend (price < EMA20), volume breakdown
            elif (close[i] < low_20[i] and 
                  close[i] < ema_20_1d[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below Donchian(20) lower or trend reversal
            if close[i] < low_20[i] or close[i] < ema_20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above Donchian(20) upper or trend reversal
            if close[i] > high_20[i] or close[i] > ema_20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals