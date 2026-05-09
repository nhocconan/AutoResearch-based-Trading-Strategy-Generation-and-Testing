#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_R1_S1_Breakout_Trend_Volume"
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
    
    # Get 4h and 1d data for multi-timeframe analysis
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's close for Camarilla calculation
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    r1_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 4
    s1_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 4
    
    # 4h trend filter: EMA34
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 4h volume filter: current volume > 1.5 * 20-period average
    vol_series_4h = pd.Series(df_4h['volume'].values)
    vol_ma_4h = vol_series_4h.rolling(window=20, min_periods=20).mean().values
    volume_filter_4h = df_4h['volume'].values > (vol_ma_4h * 1.5)
    
    # Align all to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_4h_1h = align_htf_to_ltf(prices, df_4h, ema34_4h)
    volume_filter_1h = align_htf_to_ltf(prices, df_4h, volume_filter_4h)
    
    # Session filter: 08-20 UTC (pre-market to NY close)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or
            np.isnan(ema34_4h_1h[i]) or np.isnan(volume_filter_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_1h[i]
        s1_val = s1_1h[i]
        trend = ema34_4h_1h[i]
        vol_filter = volume_filter_1h[i]
        
        if position == 0:
            # Enter long: break above R1 with volume and above 4h trend
            if close[i] > r1_val and close[i] > trend and vol_filter:
                signals[i] = 0.20
                position = 1
            # Enter short: break below S1 with volume and below 4h trend
            elif close[i] < s1_val and close[i] < trend and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 (mean reversion to center)
            if close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: close above R1 (mean reversion to center)
            if close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals