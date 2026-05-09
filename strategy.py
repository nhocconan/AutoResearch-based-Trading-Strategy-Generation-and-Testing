#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate EMA21 on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1 levels: (H-L)*1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1_level = close_1d + camarilla_range * 1
    s1_level = close_1d - camarilla_range * 1
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Volume spike filter: current volume > 1.5 * 24-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 24)  # Need enough data for EMA21 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema21 = ema21_4h_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol_spike = volume_spike[i]
        in_session = session_filter[i]
        
        if position == 0:
            # Enter long: Close breaks above R1 + 4h uptrend + volume spike + session
            if close[i] > r1 and close[i] > ema21 and vol_spike and in_session:
                signals[i] = 0.20
                position = 1
            # Enter short: Close breaks below S1 + 4h downtrend + volume spike + session
            elif close[i] < s1 and close[i] < ema21 and vol_spike and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below S1 or 4h trend turns down
            if close[i] < s1 or close[i] < ema21:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Close rises above R1 or 4h trend turns up
            if close[i] > r1 or close[i] > ema21:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals