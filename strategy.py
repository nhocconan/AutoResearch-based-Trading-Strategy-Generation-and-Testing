#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot (R1/S1) breakout with 1w EMA50 trend filter and volume confirmation
# Camarilla pivots identify key intraday support/resistance levels from prior day
# Breakout above R1 with uptrend 1w EMA50 + volume spike = long entry
# Breakdown below S1 with downtrend 1w EMA50 + volume spike = short entry
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)
# Discrete sizing 0.25 targets 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_Camarilla_R1S1_Breakout_1wEMA50_VolumeSpike"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter from prior completed 1w bar
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_shifted = np.roll(ema50_1w, 1)
    ema50_1w_shifted[0] = np.nan
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Start from 2 to ensure we have prior day's data
        # Skip if any value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(close[i-1]) or 
            np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivots from prior day (i-1)
        high_prev = high[i-1]
        low_prev = low[i-1]
        close_prev = close[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        camarilla_multiplier = 1.1 / 12
        R1 = close_prev + camarilla_multiplier * range_prev
        S1 = close_prev - camarilla_multiplier * range_prev
        
        # Volume confirmation: volume > 1.5 * 20-period EMA of volume
        if i >= 20:
            vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
            volume_threshold = 1.5 * vol_ema_20
        else:
            volume_threshold = np.inf  # No volume filter until we have enough data
        
        if position == 0:
            # Long conditions: breakout above R1 AND 1w EMA50 uptrend AND volume spike
            if close[i] > R1 and close[i] > ema50_1w_aligned[i] and volume[i] > volume_threshold:
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below S1 AND 1w EMA50 downtrend AND volume spike
            elif close[i] < S1 and close[i] < ema50_1w_aligned[i] and volume[i] > volume_threshold:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below R1 (failed breakout) OR below prior day's low
            if close[i] < R1 or close[i] < low_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above S1 (failed breakdown) OR above prior day's high
            if close[i] > S1 or close[i] > high_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals