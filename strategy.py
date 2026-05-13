#!/usr/bin/env python3
name = "1h_Donchian_Breakout_Volume_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4H and 1D data ONCE
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4H Donchian breakout (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donch_high = np.full_like(high_4h, np.nan)
    donch_low = np.full_like(low_4h, np.nan)
    
    for i in range(len(high_4h)):
        if i >= 19:
            donch_high[i] = np.max(high_4h[i-19:i+1])
            donch_low[i] = np.min(low_4h[i-19:i+1])
    
    # Align Donchian levels to 1H
    donch_high_1h = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_1h = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # 1D Trend filter: EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average (1H)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(donch_high_1h[i]) or np.isnan(donch_low_1h[i]) or 
            np.isnan(ema50_1h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if not (session_filter[i] and vol_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: break above Donchian high + above 1D EMA50
            if close[i] > donch_high_1h[i] and close[i] > ema50_1h[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: break below Donchian low + below 1D EMA50
            elif close[i] < donch_low_1h[i] and close[i] < ema50_1h[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price below Donchian low or below 1D EMA50
            if close[i] < donch_low_1h[i] or close[i] < ema50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price above Donchian high or above 1D EMA50
            if close[i] > donch_high_1h[i] or close[i] > ema50_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals