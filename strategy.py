#!/usr/bin/env python3
"""
4h_Pivot_R1S1_R2S2_Breakout_12hEMA34_Volume
Pivot breakout system using 1d Camarilla pivot levels + 12h EMA34 trend + volume confirmation.
- Long when price breaks above R1 with volume > 1.5x MA(20) and price > 12h EMA34
- Short when price breaks below S1 with volume > 1.5x MA(20) and price < 12h EMA34
- Exit when price crosses back through pivot point (PP)
- Designed for 20-40 trades/year per symbol
Works in bull markets (breaks R1/R2) and bear markets (breaks S1/S2)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for given period."""
    typical = (high + low + close) / 3
    range_val = high - low
    
    pp = typical
    r1 = close + (range_val * 1.1 / 12)
    s1 = close - (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    s2 = close - (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    s3 = close - (range_val * 1.1 / 4)
    r4 = close + (range_val * 1.1 / 2)
    s4 = close - (range_val * 1.1 / 2)
    
    return pp, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivots
    pp_1d, r1_1d, r2_1d, r3_1d, r4_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla_pivot(high_1d, low_1d, close_1d)
    
    # Align 1d pivots to 4h timeframe
    pp_1d_4h = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_4h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s1_1d_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_4h = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 4h timeframe
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_4h[i]) or np.isnan(r1_1d_4h[i]) or np.isnan(s1_1d_4h[i]) or
            np.isnan(ema_34_12h_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition
        vol_ok = volume[i] > vol_threshold[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume + above 12h EMA34
            if close[i] > r1_1d_4h[i] and vol_ok and close[i] > ema_34_12h_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + below 12h EMA34
            elif close[i] < s1_1d_4h[i] and vol_ok and close[i] < ema_34_12h_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if close[i] < pp_1d_4h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if close[i] > pp_1d_4h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_R2S2_Breakout_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0