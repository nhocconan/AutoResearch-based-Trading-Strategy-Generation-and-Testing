#!/usr/bin/env python3
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
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Elder Ray components on daily
    ema13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray to 6h
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 60-period EMA for trend filter (approx 15 days)
    ema60_6h = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume filter: above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC (most active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(ema60_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below 60-period EMA
        uptrend = close[i] > ema60_6h[i]
        downtrend = close[i] < ema60_6h[i]
        
        # Elder Ray signals:
        # Long: bull power positive AND increasing in uptrend
        # Short: bear power negative AND decreasing in downtrend
        if i > 0:
            bull_rising = bull_power_6h[i] > bull_power_6h[i-1]
            bear_falling = bear_power_6h[i] < bear_power_6h[i-1]
        else:
            bull_rising = False
            bear_falling = False
        
        long_entry = bull_power_6h[i] > 0 and bull_rising and uptrend and vol_filter
        short_entry = bear_power_6h[i] < 0 and bear_falling and downtrend and vol_filter
        
        # Exit when Elder Ray signal weakens or trend changes
        long_exit = (bull_power_6h[i] <= 0) or (not uptrend)
        short_exit = (bear_power_6h[i] >= 0) or (not downtrend)
        
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

name = "6h_ElderRay_Trend_Volume_Session"
timeframe = "6h"
leverage = 1.0