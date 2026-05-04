#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 4h EMA50 for trend direction and 1h Camarilla levels for precise entry timing
# Volume spike (2x 20-bar EMA) confirms breakout strength
# Session filter (08-20 UTC) reduces noise trades
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Works in bull/bear markets via 4h trend filter + volume confirmation

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_Session"
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1h data for Camarilla pivot calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate Camarilla levels: R1 = Close + 1.1*(High-Low)/12, S1 = Close - 1.1*(High-Low)/12
    camarilla_range = high_1h - low_1h
    r1 = close_1h + (1.1 * camarilla_range / 12)
    s1 = close_1h - (1.1 * camarilla_range / 12)
    
    # Align to 1h timeframe (wait for completed 1h bar)
    r1_aligned = align_htf_to_ltf(prices, df_1h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1h, s1)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R1 AND 4h EMA50 uptrend AND volume spike
            if close[i] > r1_aligned[i] and close[i] > ema50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: break below S1 AND 4h EMA50 downtrend AND volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema50_4h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below R1 OR below 4h EMA50
            if close[i] < r1_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above S1 OR above 4h EMA50
            if close[i] > s1_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals