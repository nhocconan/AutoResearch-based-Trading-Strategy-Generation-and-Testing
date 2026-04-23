#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
- Uses 4h EMA(50) for trend direction (only long when price > EMA, short when price < EMA)
- 1h Camarilla pivot levels (R1, S1) from 1h timeframe for precise breakout entries
- Volume confirmation (> 1.5x 20-period average) filters low-momentum breakouts
- Session filter: only trade 08:00-20:00 UTC to avoid low-volume Asian session noise
- Designed for 1h timeframe targeting 15-37 trades/year (60-150 over 4 years)
- Uses HTF (4h) for signal direction, 1h only for entry timing to reduce overtrading
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
    open_time = prices['open_time']
    
    # Pre-compute session hours (08:00-20:00 UTC) for filter
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA(50) for trend filter (HTF for direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate 1h Camarilla pivot levels (R1, S1) - based on previous 1h bar
    # For 1h timeframe, we use the previous bar's OHLC
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    # Set first value to NaN since no previous bar
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_1h = high_prev - low_prev
    r1 = pivot + range_1h * 1.1 / 4
    s1 = pivot - range_1h * 1.1 / 4
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 1)  # EMA50, volume MA, and we need previous bar for Camarilla
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions using current price vs previous bar's Camarilla levels
        price_above_r1 = close[i] > r1[i]
        price_below_s1 = close[i] < s1[i]
        
        # Trend filter from 4h EMA: price > EMA for long, price < EMA for short
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