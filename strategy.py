#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
- Uses Camarilla pivot levels (R1, S1) from 4h timeframe for breakout signals
- 4h EMA(50) defines trend direction (only long when price > EMA, short when price < EMA)
- Volume confirmation (> 1.5x 20-period average) filters low-momentum breakouts
- Session filter (08-20 UTC) reduces noise trades outside active hours
- Target: 60-150 total trades over 4 years = 15-37/year for 1h
- Works in both bull and bear markets by trading with the 4h trend
- Discrete position sizing: 0.20 (20% of capital) to manage drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate 4h Camarilla pivot levels (R1, S1)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla pivot calculation: based on previous 4h bar's OHLC
    pivot = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1 = pivot + range_4h * 1.1 / 4
    s1 = pivot - range_4h * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Calculate 4h EMA(50) for trend filter
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above R1, uptrend, volume spike, in session
            long_signal = (price_above_r1 and 
                          uptrend and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below S1, downtrend, volume spike, in session
            short_signal = (price_below_s1 and 
                           downtrend and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: opposite Camarilla breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S1 or trend turns down
                if (price_below_s1 or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R1 or trend turns up
                if (price_above_r1 or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0