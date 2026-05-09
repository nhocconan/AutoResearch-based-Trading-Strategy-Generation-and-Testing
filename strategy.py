#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
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
    
    # Get 4h data for Camarilla levels and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's close for Camarilla calculation (R1, S1) - using 4h data to simulate daily
    # We'll use 4h data to calculate daily levels by resampling conceptually but using actual 4h bars
    # For simplicity, we use previous 4h bar's high/low/close to calculate intraday Camarilla
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Calculate Camarilla levels (R1, S1) - using 1/6 and 5/6 multipliers for inner bands
    r1 = prev_close + 1.1 * (prev_high - prev_low) * 1 / 6
    s1 = prev_close - 1.1 * (prev_high - prev_low) * 1 / 6
    
    # Trend filter: 4h EMA20
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current 1d volume > 1.3 * 20-period average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.3)
    
    # Align all to 1h
    r1_1h = align_htf_to_ltf(prices, df_4h, r1)
    s1_1h = align_htf_to_ltf(prices, df_4h, s1)
    ema20_4h_1h = align_htf_to_ltf(prices, df_4h, ema20_4h)
    volume_filter_1d_1h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)  # Need enough data for EMA20 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or
            np.isnan(ema20_4h_1h[i]) or np.isnan(volume_filter_1d_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_1h[i]
        s1_val = s1_1h[i]
        trend = ema20_4h_1h[i]
        vol_filter = volume_filter_1d_1h[i]
        session_ok = in_session[i]
        
        if position == 0:
            # Enter long: break above R1 with volume, above trend, and in session
            if close[i] > r1_val and close[i] > trend and vol_filter and session_ok:
                signals[i] = 0.20
                position = 1
            # Enter short: break below S1 with volume, below trend, and in session
            elif close[i] < s1_val and close[i] < trend and vol_filter and session_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 (mean reversion to center) or out of session
            if close[i] < s1_val or not session_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: close above R1 (mean reversion to center) or out of session
            if close[i] > r1_val or not session_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals