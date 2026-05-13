#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1D_Trend_v2"
timeframe = "4h"
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
    
    # Load 1D data ONCE for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from 1D
    pivot = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3
    r1 = close_1d[-1] + (high_1d[-1] - low_1d[-1]) * 1.1 / 12
    s1 = close_1d[-1] - (high_1d[-1] - low_1d[-1]) * 1.1 / 12
    
    # Arrays for 1D levels (same value for all 4h bars within the day)
    r1_1d = np.full_like(close_1d, r1)
    s1_1d = np.full_like(close_1d, s1)
    pivot_1d = np.full_like(close_1d, pivot)
    
    # Calculate EMA34 on 1D for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1D indicators to 4H timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection on 4H
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(pivot_4h[i]) or np.isnan(ema34_4h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA34
        price_above_ema = close[i] > ema34_4h[i]
        price_below_ema = close[i] < ema34_4h[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and uptrend
            if close[i] > r1_4h[i] and vol_spike[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and downtrend
            elif close[i] < s1_4h[i] and vol_spike[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot or trend weakens
            if close[i] < pivot_4h[i] or close[i] < ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot or trend weakens
            if close[i] > pivot_4h[i] or close[i] > ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals