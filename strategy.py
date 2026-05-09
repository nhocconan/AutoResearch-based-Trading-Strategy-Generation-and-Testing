#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w timeframe for direction and structure.
# Uses 1w Donchian(20) breakouts with 1w EMA20 trend filter and volume confirmation.
# Designed to work in both bull and bear markets by requiring alignment with weekly trend.
# Target: 10-30 trades per year to minimize fee drift and improve generalization.
name = "1d_Donchian20_1wEMA20_VolumeBreakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian and EMA20
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1w Donchian(20) channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: 20-period high, Lower band: 20-period low
    upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1w Donchian levels to 1d timeframe
    upper_1d = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1d = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Volume filter: spike above 2.0x 10-period average
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Wait for EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1d[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(upper_1d[i]) or np.isnan(lower_1d[i])):
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
            # Long: price above 1w upper band, 1w uptrend (price > EMA20), volume breakout
            if (close[i] > upper_1d[i] and 
                close[i] > ema_20_1d[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below 1w lower band, 1w downtrend (price < EMA20), volume breakdown
            elif (close[i] < lower_1d[i] and 
                  close[i] < ema_20_1d[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 1w lower band or trend reversal
            if close[i] < lower_1d[i] or close[i] < ema_20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 1w upper band or trend reversal
            if close[i] > upper_1d[i] or close[i] > ema_20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals