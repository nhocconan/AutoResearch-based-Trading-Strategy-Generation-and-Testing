#!/usr/bin/env python3
name = "1h_Camarilla_R1S1_Breakout_Trend_4h1d"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for direction (trend and Camarilla)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h pivot and Camarilla levels
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    r1_4h = close_4h + (range_4h * 1.0833)
    s1_4h = close_4h - (range_4h * 1.0833)
    
    # Align Camarilla levels to 1h
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA34 for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 1h volume > 1.5x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if session_filter[i] and volume_filter[i]:
            if position == 0:
                # Long: Close above R1, above 4h EMA50, above 1d EMA34
                if (close[i] > r1_4h_aligned[i] and 
                    close[i] > ema_50_4h_aligned[i] and 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.20
                    position = 1
                # Short: Close below S1, below 4h EMA50, below 1d EMA34
                elif (close[i] < s1_4h_aligned[i] and 
                      close[i] < ema_50_4h_aligned[i] and 
                      close[i] < ema_34_1d_aligned[i]):
                    signals[i] = -0.20
                    position = -1
            elif position == 1:
                # Exit long: Close below S1 or below 4h EMA50
                if (close[i] < s1_4h_aligned[i] or 
                    close[i] < ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: Close above R1 or above 4h EMA50
                if (close[i] > r1_4h_aligned[i] or 
                    close[i] > ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # Outside session or no volume confirmation: flatten
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals