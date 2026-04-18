# 6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v2
# Hypothesis: Use daily Camarilla R1/S1 levels with volume confirmation and ATR-based volatility filter
# to capture breakouts with follow-through. Works in bull and bear markets by avoiding low volatility
# conditions and only trading when price breaks key daily support/resistance with volume.
# Targets 12-37 trades/year with position size 0.25.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = Close + 1.1 * (High - Low)
    # S1 = Close - 1.1 * (High - Low)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h timeframe (wait for daily bar close)
    r1_6h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate ATR (14-period) for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr14 = np.full(n, np.nan)
    for i in range(14, n):
        atr14[i] = np.mean(tr[i-13:i+1])
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime filter)
    atr50 = np.full(n, np.nan)
    for i in range(50, n):
        atr50[i] = np.mean(atr14[i-49:i+1])
    atr_ratio = np.full(n, np.nan)
    for i in range(50, n):
        if atr50[i] > 0:
            atr_ratio[i] = atr14[i] / atr50[i]
        else:
            atr_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # need volume MA and ATR ratio
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter: only trade when ATR ratio > 0.8 (avoid low volatility)
        vol_filter = atr_ratio[i] > 0.8
        
        if position == 0:
            # Long entry: price breaks above S1 with volume confirmation and volatility filter
            if close[i] > s1_6h[i] and vol_confirmed and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below R1 with volume confirmation and volatility filter
            elif close[i] < r1_6h[i] and vol_confirmed and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses back below S1
            if close[i] < s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above R1
            if close[i] > r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v2"
timeframe = "6h"
leverage = 1.0