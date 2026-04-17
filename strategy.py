#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla Pivot R1/S1 Breakout + 1w Trend Filter + Volume Confirmation.
Long when price breaks above R1 with 1w EMA50 uptrend and volume > 1.5x average.
Short when price breaks below S1 with 1w EMA50 downtrend and volume > 1.5x average.
Exit when price reverts to pivot point (PP) or opposite Camarilla level (S1/R1).
Uses 1d for Camarilla levels and volume, 1w for EMA50 trend filter.
Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
"""

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
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    # PP = (High + Low + Close)/3
    rng = high_1d - low_1d
    r1_1d = close_1d + 1.1 * rng / 12
    s1_1d = close_1d - 1.1 * rng / 12
    pp_1d = (high_1d + low_1d + close_1d) / 3
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Align 1w EMA50
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate average volume (20-day) for volume spike filter
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price and volume conditions
        price = close[i]
        vol = volume_1d_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        vol_spike = vol > 1.5 * vol_ma if vol_ma > 0 else False
        
        # Trend filter from 1w EMA50
        ema50 = ema50_1w_aligned[i]
        is_uptrend = price > ema50
        is_downtrend = price < ema50
        
        # Camarilla levels
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        pp = pp_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with uptrend and volume spike
            if price > r1 and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with downtrend and volume spike
            elif price < s1 and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to PP or breaks below S1 (reversal)
            if price <= pp or price < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to PP or breaks above R1 (reversal)
            if price >= pp or price > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0