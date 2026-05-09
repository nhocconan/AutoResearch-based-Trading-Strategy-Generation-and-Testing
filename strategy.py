#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h and 1d timeframes for direction, 1h for entry timing.
# Uses 4h Donchian channel breakout (20-period) with 1d EMA50 trend filter and volume confirmation.
# Designed to work in both bull and bear markets by requiring alignment with higher timeframe trend.
# Target: 15-37 trades per year to minimize fee drag and improve generalization.
name = "1h_Donchian20_1dEMA50_VolumeBreakout"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channel (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high of last 20 periods
    upper_4h = np.full(len(high_4h), np.nan)
    for i in range(20, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i-20:i])
    
    # Lower band: lowest low of last 20 periods
    lower_4h = np.full(len(low_4h), np.nan)
    for i in range(20, len(low_4h)):
        lower_4h[i] = np.min(low_4h[i-20:i])
    
    # Align 4h Donchian to 1h timeframe
    upper_4h_1h = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_1h = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: spike above 2.0x 24-period average (1 day of 1h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24, 20*4)  # Wait for EMA50, volume MA, and 4h Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(upper_4h_1h[i]) or np.isnan(lower_4h_1h[i])):
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
            # Long: price above 4h upper band, 1d uptrend (price > EMA50), volume breakout
            if (close[i] > upper_4h_1h[i] and 
                close[i] > ema_50_1h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: price below 4h lower band, 1d downtrend (price < EMA50), volume breakdown
            elif (close[i] < lower_4h_1h[i] and 
                  close[i] < ema_50_1h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price below 4h lower band or trend reversal
            if close[i] < lower_4h_1h[i] or close[i] < ema_50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price above 4h upper band or trend reversal
            if close[i] > upper_4h_1h[i] or close[i] > ema_50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals