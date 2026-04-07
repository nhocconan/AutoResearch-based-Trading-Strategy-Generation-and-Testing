#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: Fade at Camarilla R3/S3 levels and breakout continuation at R4/S4 levels on 6h timeframe, 
filtered by 1-day EMA50 trend and volume confirmation. Designed for mean reversion at extreme levels 
and breakout continuation with proper risk control. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1-day data
    # Using previous day's high, low, close (already available in df_1d)
    ph = df_1d['high'].values  # previous day high
    pl = df_1d['low'].values   # previous day low
    pc = df_1d['close'].values # previous day close
    
    # Camarilla levels: H/L = high-low, C = close
    hl = ph - pl
    r3 = pc + (hl * 1.1 / 4)  # Resistance 3
    s3 = pc - (hl * 1.1 / 4)  # Support 3
    r4 = pc + (hl * 1.1 / 2)  # Resistance 4
    s4 = pc - (hl * 1.1 / 2)  # Support 4
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(pc).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 24-period average (4 days of 6h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Price levels
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R4 (take profit) or breaks below S3 with volume (stop)
            if close[i] >= r4_val or (close[i] < s3_val and vol_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S4 (take profit) or breaks above R3 with volume (stop)
            if close[i] <= s4_val or (close[i] > r3_val and vol_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion long at S3: price touches/bounces off S3 with volume
            if close[i] <= s3_val and vol_confirmed and close[i] > ema50_val:
                position = 1
                signals[i] = 0.25
            # Mean reversion short at R3: price touches/bounces off R3 with volume
            elif close[i] >= r3_val and vol_confirmed and close[i] < ema50_val:
                position = -1
                signals[i] = -0.25
            # Breakout long: price breaks above R4 with volume and above EMA50
            elif close[i] > r4_val and vol_confirmed and close[i] > ema50_val:
                position = 1
                signals[i] = 0.25
            # Breakout short: price breaks below S4 with volume and below EMA50
            elif close[i] < s4_val and vol_confirmed and close[i] < ema50_val:
                position = -1
                signals[i] = -0.25
    
    return signals