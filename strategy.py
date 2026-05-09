#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1S1_Breakout_4hTrend_DailyVol"
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
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (based on previous day)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + (range_hl * 1.1 / 4)
    R2 = pivot + (range_hl * 1.1 / 2)
    S1 = pivot - (range_hl * 1.1 / 4)
    S2 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 1h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # Trend filter: 4h EMA20
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume filter: current 1h volume > 2.0 * 24-period average (daily average)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(24, 20)  # Need enough data for volume MA and EMA20
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(ema20_4h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1 = R1_aligned[i]
        r2 = R2_aligned[i]
        s1 = S1_aligned[i]
        s2 = S2_aligned[i]
        trend = ema20_4h_aligned[i]
        vol_filter = volume_filter[i]
        in_session = session_filter[i]
        
        if position == 0:
            # Enter long: break above R1 with volume, above trend, and in session
            if close[i] > r1 and close[i] > trend and vol_filter and in_session:
                signals[i] = 0.20
                position = 1
            # Enter short: break below S1 with volume, below trend, and in session
            elif close[i] < s1 and close[i] < trend and vol_filter and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 (reversion to mean)
            if close[i] < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: close above R1 (reversion to mean)
            if close[i] > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals