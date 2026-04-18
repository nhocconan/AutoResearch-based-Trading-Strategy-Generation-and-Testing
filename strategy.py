#!/usr/bin/env python3
"""
12h_MultiTF_Pivot_R1S1_TrendFilter_V2
12h strategy using daily Camarilla pivot levels with 1h trend filter and volume confirmation.
- Long: Close > R1 + 1h EMA34 > EMA89 + volume > 1.5x daily avg
- Short: Close < S1 + 1h EMA34 < EMA89 + volume > 1.5x daily avg
- Exit: Opposite pivot breach or trend change
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (trend continuation) and bear markets (trend reversal at pivots)
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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align daily pivot levels to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 1h data for trend filter
    df_1h = get_htf_data(prices, '1h')
    
    close_1h = df_1h['close'].values
    
    # 1h EMA34 and EMA89 for trend filter
    ema_34_1h = pd.Series(close_1h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1h = pd.Series(close_1h).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    ema_34_aligned = align_htf_to_ltf(prices, df_1h, ema_34_1h)
    ema_89_aligned = align_htf_to_ltf(prices, df_1h, ema_89_1h)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 89  # need enough for EMA89
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_89_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_34_aligned[i] > ema_89_aligned[i]
        downtrend = ema_34_aligned[i] < ema_89_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Pivot level conditions
        above_r1 = close[i] > r1_aligned[i]
        below_s1 = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + price above R1
            if uptrend and vol_confirm and above_r1:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + price below S1
            elif downtrend and vol_confirm and below_s1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volume confirmation, or price below S1
            if not uptrend or (vol_confirm and below_s1):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volume confirmation, or price above R1
            if not downtrend or (vol_confirm and above_r1):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_MultiTF_Pivot_R1S1_TrendFilter_V2"
timeframe = "12h"
leverage = 1.0