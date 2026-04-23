#!/usr/bin/env python3
"""
Hypothesis: 1h 4-hr Camarilla R1/S1 breakout with volume spike and session filter (08-20 UTC).
Long when price breaks above R1 with volume > 2x average and in session.
Short when price breaks below S1 with volume > 2x average and in session.
Exit when price returns to Camarilla pivot (PP) or volume drops below average.
Uses 4h Camarilla levels for structure and 1h for precise entry timing.
Designed to generate 60-150 total trades over 4 years with low frequency to minimize fee drag.
Works in both bull and bear markets by trading breakouts from key intraday levels.
"""

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
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data for Camarilla levels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for 4h: R1, S1, PP
    # Camarilla: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    pp_4h = (high_4h + low_4h + close_4h) / 3.0
    r1_4h = close_4h + (high_4h - low_4h) * 1.1 / 12.0
    s1_4h = close_4h - (high_4h - low_4h) * 1.1 / 12.0
    
    # Align 4h Camarilla levels to 1h timeframe (wait for completed 4h bar)
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume average (24-period ~1 day) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(pp_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = pp_4h_aligned[i]
        r1_val = r1_4h_aligned[i]
        s1_val = s1_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike
            if price > r1_val and vol_current > 2.0 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume spike
            elif price < s1_val and vol_current > 2.0 * vol_ma_val:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to pivot PP
                if price <= pp_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to pivot PP
                if price >= pp_val:
                    exit_signal = True
            
            # Also exit if volume drops below average (loss of momentum)
            if vol_current < vol_ma_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_Volume_Session"
timeframe = "1h"
leverage = 1.0