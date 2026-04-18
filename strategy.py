#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Trend_Filter
Hypothesis: Breakout above Camarilla R1 or below S1 with volume confirmation and 1-day EMA trend filter captures institutional momentum while avoiding false breakouts. Camarilla levels act as high-probability support/resistance, volume confirms institutional participation, and daily trend filter ensures alignment with higher timeframe momentum. Works in both bull (breakouts continue) and bear (breakdowns continue) markets. Target: 20-40 trades/year (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # Need daily OHLC from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for current day based on previous day
    # R1 = close + (high - low) * 1.12 / 12
    # S1 = close - (high - low) * 1.12 / 12
    rng = prev_high - prev_low
    r1 = prev_close + rng * 1.12 / 12
    s1 = prev_close - rng * 1.12 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1-day EMA trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_1d_4h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_4h[i]
        s1_val = s1_4h[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_4h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend
            if price > r1_val and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in downtrend
            elif price < s1_val and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long until price crosses below S1 or trend reverses
            if price < s1_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Maintain short until price crosses above R1 or trend reverses
            if price > r1_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0