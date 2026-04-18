#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume
4h strategy based on Camarilla pivot levels from 1d:
- Long: Close > R1 and price > S1 + volume > 1.5x average + 1d EMA34 > EMA89
- Short: Close < S1 and price < R1 + volume > 1.5x average + 1d EMA34 < EMA89
- Exit: Opposite signal or trend reversal
Designed for ~20-50 trades/year per symbol (80-200 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
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
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Resistance and Support levels
    R1 = pivot + (range_hl * 1.1 / 12)
    S1 = pivot - (range_hl * 1.1 / 12)
    R2 = pivot + (range_hl * 1.1 / 6)
    S2 = pivot - (range_hl * 1.1 / 6)
    
    # Align Camarilla levels to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # Daily EMA34 and EMA89 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 89  # need enough for EMA89
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_89_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_34_aligned[i] > ema_89_aligned[i]
        downtrend = ema_34_aligned[i] < ema_89_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Price relative to Camarilla levels
        price_above_S1 = close[i] > S1_aligned[i]
        price_below_R1 = close[i] < R1_aligned[i]
        price_below_S2 = close[i] < S2_aligned[i]
        price_above_R2 = close[i] > R2_aligned[i]
        
        if position == 0:
            # Long: price above S1, in uptrend, volume confirmation, and breaks above R1
            if price_above_S1 and uptrend and vol_confirm and close[i] > R1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below R1, in downtrend, volume confirmation, and breaks below S1
            elif price_below_R1 and downtrend and vol_confirm and close[i] < S1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or price breaks below S2
            if not uptrend or close[i] < S2_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or price breaks above R2
            if not downtrend or close[i] > R2_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0