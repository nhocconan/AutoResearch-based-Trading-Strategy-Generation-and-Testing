#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Camarilla_R1S1_Breakout_Trend_Volume"
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
    
    # Get 4h data for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h close for EMA trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 4h high/low/close for Camarilla levels (using previous day's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for 4h (using previous 4h bar)
    # R1 = Close + 1.1*(High - Low)/12
    # S1 = Close - 1.1*(High - Low)/12
    rang = high_4h - low_4h
    r1_4h = close_4h + 1.1 * rang / 12
    s1_4h = close_4h - 1.1 * rang / 12
    
    # Align Camarilla levels to 1h
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # 1d volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    vol_ratio = vol_1d / np.where(vol_ma_20_1d_aligned > 0, vol_ma_20_1d_aligned, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + above 4h EMA20 + volume spike
            if (close[i] > r1_4h_aligned[i] and 
                close[i] > ema_20_4h_aligned[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + below 4h EMA20 + volume spike
            elif (close[i] < s1_4h_aligned[i] and 
                  close[i] < ema_20_4h_aligned[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price below S1 OR below 4h EMA20
            if close[i] < s1_4h_aligned[i] or close[i] < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price above R1 OR above 4h EMA20
            if close[i] > r1_4h_aligned[i] or close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals