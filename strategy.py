#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
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
    
    # Get 4h data for trend and volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's close for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R1, S1)
    r1 = prev_close + 1.1 * (prev_high - prev_low) / 4
    s1 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Trend filter: 4h EMA50
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 4h volume > 1.5 * 20-day average
    vol_series = pd.Series(df_4h['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_4h = df_4h['volume'].values > (vol_ma * 1.5)
    
    # Align all to 1h
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    ema50_4h_1h = align_htf_to_ltf(prices, df_4h, ema50_4h)
    volume_filter_1h = align_htf_to_ltf(prices, df_4h, volume_filter_4h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or
            np.isnan(ema50_4h_1h[i]) or np.isnan(volume_filter_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_1h[i]
        s1_val = s1_1h[i]
        trend = ema50_4h_1h[i]
        vol_filter = volume_filter_1h[i]
        
        if position == 0:
            # Enter long: break above R1 with volume and above trend
            if close[i] > r1_val and close[i] > trend and vol_filter:
                signals[i] = 0.20
                position = 1
            # Enter short: break below S1 with volume and below trend
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