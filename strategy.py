#!/usr/bin/env python3
"""
6h Weekly Pivot Reversal with Volume Confirmation.
Fade at weekly R3/S3 levels during low volatility, breakout at R4/S4 during high volatility.
Uses weekly pivot levels calculated from prior week's high/low/close.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_reversal_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly pivot levels (from prior week) ===
    # Get weekly data
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivots for previous week (shifted by 1 to avoid look-ahead)
    pp = (weekly_high[:-1] + weekly_low[:-1] + weekly_close[:-1]) / 3
    r1 = 2 * pp - weekly_low[:-1]
    s1 = 2 * pp - weekly_high[:-1]
    r2 = pp + (weekly_high[:-1] - weekly_low[:-1])
    s2 = pp - (weekly_high[:-1] - weekly_low[:-1])
    r3 = weekly_high[:-1] + 2 * (pp - weekly_low[:-1])
    s3 = weekly_low[:-1] - 2 * (weekly_high[:-1] - pp)
    r4 = r3 + (weekly_high[:-1] - weekly_low[:-1])
    s4 = s3 - (weekly_high[:-1] - weekly_low[:-1])
    
    # Align to 6s timeframe (shifted by 1 week already in calculation)
    pp_aligned = align_htf_to_ltf(prices, df_weekly[:-1], pp)  # exclude last incomplete week
    r3_aligned = align_htf_to_ltf(prices, df_weekly[:-1], r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly[:-1], s3)
    r4_aligned = align_htf_to_ltf(prices, df_weekly[:-1], r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly[:-1], s4)
    
    # === Volatility filter (ATR-based) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R3 or R4 (take profit) or crosses S3 (stop)
            if close[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S3 or S4 (take profit) or crosses R3 (stop)
            if close[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need above average volume
            if vol_ratio[i] < 1.1:
                signals[i] = 0.0
                continue
            
            # High volatility breakout mode (ATR above average)
            if atr[i] > np.nanmedian(atr[max(0, i-50):i+1]):
                # Breakout mode: break R4 or S4
                if close[i] > r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            else:
                # Low volatility mean reversion mode: fade at R3/S3
                if close[i] < r3_aligned[i] and close[i] > s3_aligned[i]:
                    # In the range between S3 and R3
                    if close[i] <= s3_aligned[i] + (r3_aligned[i] - s3_aligned[i]) * 0.3:
                        # Near S3, look for long
                        if close[i] > close[i-1]:  # Confirm with uptick
                            position = 1
                            signals[i] = 0.25
                    elif close[i] >= r3_aligned[i] - (r3_aligned[i] - s3_aligned[i]) * 0.3:
                        # Near R3, look for short
                        if close[i] < close[i-1]:  # Confirm with downtick
                            position = -1
                            signals[i] = -0.25
    
    return signals