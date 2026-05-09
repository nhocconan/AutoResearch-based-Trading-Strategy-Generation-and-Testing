#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
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
    
    # Get 4h data for trend and Camarilla parameters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA for trend
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 4h ATR for Camarilla levels
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = abs(df_4h['high'] - df_4h['close'].shift())
    tr3 = abs(df_4h['low'] - df_4h['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate 4h close for Camarilla pivot
    close_4h = df_4h['close'].values
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    # Camarilla levels: P = close, Range = high - low
    # R1 = P + 1.1 * Range / 12, S1 = P - 1.1 * Range / 12
    range_4h = df_4h['high'] - df_4h['low']
    r1_4h = close_4h + 1.1 * range_4h / 12
    s1_4h = close_4h - 1.1 * range_4h / 12
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume filter: current 1h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or
            np.isnan(r1_4h_aligned[i]) or
            np.isnan(s1_4h_aligned[i]) or
            np.isnan(volume_filter[i]) or
            np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_4h_aligned[i]
        r1 = r1_4h_aligned[i]
        s1 = s1_4h_aligned[i]
        vol_filter = volume_filter[i]
        sess_filter = session_filter[i]
        
        if position == 0:
            # Enter long: close above R1 + above 4h EMA trend + volume + session
            if close[i] > r1 and close[i] > ema50_val and vol_filter and sess_filter:
                signals[i] = 0.20
                position = 1
            # Enter short: close below S1 + below 4h EMA trend + volume + session
            elif close[i] < s1 and close[i] < ema50_val and vol_filter and sess_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: close below 4h EMA trend
            if close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: close above 4h EMA trend
            if close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals