#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_Pullback_4hTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Daily Camarilla levels (using previous day's OHLC)
    prev_close_1d = np.roll(df_1d['close'], 1)
    prev_high_1d = np.roll(df_1d['high'], 1)
    prev_low_1d = np.roll(df_1d['low'], 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Camarilla levels: H3, L3 (tighter levels for more precise entries)
    H3 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 4
    L3 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 4
    
    # Align 4h EMA and daily Camarilla levels to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough for EMA and Camarilla
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        ema_4h = ema_4h_aligned[i]
        h3 = H3_aligned[i]
        l3 = L3_aligned[i]
        
        if position == 0 and in_session:
            # Enter long: Pullback to L3 in uptrend (price > 4h EMA20)
            if close[i] <= l3 and close[i] > ema_4h:
                signals[i] = 0.20
                position = 1
            # Enter short: Pullback to H3 in downtrend (price < 4h EMA20)
            elif close[i] >= h3 and close[i] < ema_4h:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 4h EMA20 (trend change)
            if close[i] < ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price crosses above 4h EMA20 (trend change)
            if close[i] > ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals