#!/usr/bin/env python3
"""
1h_4h1d_Trend_Follow_With_Volume_Filter_v1
1-hour strategy using 4h trend and 1d volume confirmation for entry timing.
Trades only during 08-20 UTC session to reduce noise.
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h EMA trend (21-period) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d Volume confirmation (20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(vol_1d_current[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 1.3x 20-day average
        vol_confirmed = vol_1d_current[i] > 1.3 * vol_ma_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above 4h EMA with volume confirmation
            if close[i] > ema_4h_aligned[i] and vol_confirmed:
                signals[i] = 0.20
                position = 1
                continue
            # Short: price below 4h EMA with volume confirmation
            elif close[i] < ema_4h_aligned[i] and vol_confirmed:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic: trend reversal
        elif position == 1:
            # Exit long: price crosses below 4h EMA
            if close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above 4h EMA
            if close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Trend_Follow_With_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0