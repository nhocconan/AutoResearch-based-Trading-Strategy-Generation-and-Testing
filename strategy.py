#!/usr/bin/env python3
"""
4h_4hr_VWAP_Trend_With_Volume_Filter
Hypothesis: Price above/below 4-hour VWAP with volume confirmation and 1-day EMA trend filter captures 
institutional momentum in both bull and bear markets. VWAP acts as dynamic support/resistance, 
reducing false breakouts. Target: 20-40 trades/year (80-160 total over 4 years).
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
    
    # 4-hour VWAP calculation
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # 1-day EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(vwap[i]) or np.isnan(ema_1d_4h[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_4h[i]
        
        if position == 0:
            # Long: price above VWAP with volume in uptrend
            if price > vwap_val and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP with volume in downtrend
            elif price < vwap_val and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long until price crosses below VWAP or trend reverses
            if price < vwap_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Maintain short until price crosses above VWAP or trend reverses
            if price > vwap_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4hr_VWAP_Trend_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0