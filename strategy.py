#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_Pivot_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot levels and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels: based on previous day's OHLC
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using previous 4h bar to avoid look-ahead
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot levels using previous bar
    R1 = close_4h + (high_4h - low_4h) * 1.1 / 12
    S1 = close_4h - (high_4h - low_4h) * 1.1 / 12
    
    # Shift to align with current bar (use previous bar's levels)
    R1 = np.roll(R1, 1)
    S1 = np.roll(S1, 1)
    R1[0] = np.nan
    S1[0] = np.nan
    
    # Get 4h trend: EMA20
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h indicators to 1h
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume filter: current 1h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # EMA20 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(ema20_4h_aligned[i]) or
            np.isnan(volume_filter[i]) or
            np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        ema20_val = ema20_4h_aligned[i]
        vol_filter = volume_filter[i]
        sess_filter = session_filter[i]
        
        if position == 0:
            # Enter long: close above R1 + above EMA20 + volume + session
            if close[i] > r1_val and close[i] > ema20_val and vol_filter and sess_filter:
                signals[i] = 0.20
                position = 1
            # Enter short: close below S1 + below EMA20 + volume + session
            elif close[i] < s1_val and close[i] < ema20_val and vol_filter and sess_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: close below EMA20
            if close[i] < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: close above EMA20
            if close[i] > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals