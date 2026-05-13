#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate 1-hour Camarilla levels
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Use previous hour's range for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla levels
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Load 4H data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Load 1D data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 34 EMA on 4H
        uptrend_4h = close[i] > ema34_4h_aligned[i]
        downtrend_4h = close[i] < ema34_4h_aligned[i]
        
        # Volume filter: current volume > 20-day average on 1D
        volume_filter = volume[i] > vol_ma20_1d_aligned[i]
        
        if position == 0:
            # LONG: Break above R1 + uptrend on 4H + volume confirmation
            if close[i] > R1[i] and uptrend_4h and volume_filter:
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 + downtrend on 4H + volume confirmation
            elif close[i] < S1[i] and downtrend_4h and volume_filter:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below S1 or trend reversal
            if close[i] < S1[i] or not uptrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Break above R1 or trend reversal
            if close[i] > R1[i] or not downtrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals