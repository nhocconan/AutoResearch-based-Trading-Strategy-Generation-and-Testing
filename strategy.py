#!/usr/bin/env python3
"""
4h_Pullback_Pivot_R1S1_Breakout_With_Volume_v1
Concept: 4h breakout above R1 or below S1 from daily Camarilla pivot, pulled back to pivot point.
- Long: Price breaks above R1, then pulls back to pivot point with volume confirmation
- Short: Price breaks below S1, then pulls back to pivot point with volume confirmation
- Exit: Price reaches opposite pivot level (S1 for long, R1 for short) or reverses at pivot
- Uses 12h trend filter to avoid counter-trend trades
- Position sizing: 0.25
- Target: 20-50 trades/year (80-200 total over 4 years)
- Works in bull/bear: Pivot levels adapt to volatility, volume confirms breakout strength
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Pullback_Pivot_R1S1_Breakout_With_Volume_v1"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    R1 = close + (range_val * 1.1 / 12)
    S1 = close - (range_val * 1.1 / 12)
    R2 = close + (range_val * 1.1 / 6)
    S2 = close - (range_val * 1.1 / 6)
    R3 = close + (range_val * 1.1 / 4)
    S3 = close - (range_val * 1.1 / 4)
    R4 = close + (range_val * 1.1 / 2)
    S4 = close - (range_val * 1.1 / 2)
    return pivot, R1, S1, R2, S2, R3, S3, R4, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === Daily: Camarilla pivot levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    pivot_1d = np.full_like(close_1d, np.nan)
    R1_1d = np.full_like(close_1d, np.nan)
    S1_1d = np.full_like(close_1d, np.nan)
    R2_1d = np.full_like(close_1d, np.nan)
    S2_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        p, r1, s1, r2, s2, _, _, _, _ = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        pivot_1d[i] = p
        R1_1d[i] = r1
        S1_1d[i] = s1
        R2_1d[i] = r2
        S2_1d[i] = s2
    
    # Align daily levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    
    # === 12h: EMA34 trend filter ===
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # === 4h: Volume average ===
    volume = prices['volume'].values
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    breakout_high = np.full(n, np.nan)  # Track breakout level for pullback
    breakout_low = np.full(n, np.nan)   # Track breakdown level for pullback
    
    start_idx = 30  # Ensure enough data for volume average
    
    for i in range(start_idx, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_val = volume[i]
        vol_avg_val = vol_avg[i]
        pivot_val = pivot_1d_aligned[i]
        R1_val = R1_1d_aligned[i]
        S1_val = S1_1d_aligned[i]
        R2_val = R2_1d_aligned[i]
        S2_val = S2_1d_aligned[i]
        ema34_val = ema34_12h_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(vol_val) or np.isnan(vol_avg_val) or 
            np.isnan(pivot_val) or np.isnan(R1_val) or np.isnan(S1_val) or 
            np.isnan(R2_val) or np.isnan(S2_val) or np.isnan(ema34_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
                breakout_high[i] = np.nan
                breakout_low[i] = np.nan
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol_val > 1.5 * vol_avg_val
        
        if position == 0:
            # Check for bullish breakout above R1 with pullback to pivot
            if (close_val > R1_val and 
                vol_confirm and 
                close_val > ema34_val):  # 12h uptrend filter
                # Mark breakout level and look for pullback
                breakout_high[i] = R1_val
                # If already at or near pivot, enter immediately
                if abs(close_val - pivot_val) / pivot_val < 0.005:  # Within 0.5% of pivot
                    signals[i] = 0.25
                    position = 1
            
            # Check for bearish breakdown below S1 with pullback to pivot
            elif (close_val < S1_val and 
                  vol_confirm and 
                  close_val < ema34_val):  # 12h downtrend filter
                # Mark breakdown level and look for pullback
                breakout_low[i] = S1_val
                # If already at or near pivot, enter immediately
                if abs(close_val - pivot_val) / pivot_val < 0.005:  # Within 0.5% of pivot
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long: waiting for pullback to pivot after R1 breakout
            if not np.isnan(breakout_high[i-1]):
                # Still waiting for pullback
                if abs(close_val - pivot_val) / pivot_val < 0.01:  # Within 1% of pivot
                    signals[i] = 0.25
                else:
                    signals[i] = 0.0
                    position = 0
                    breakout_high[i] = np.nan
            else:
                # No active breakout, look for new setup
                signals[i] = 0.0
                if (close_val > R1_val and 
                    vol_confirm and 
                    close_val > ema34_val):
                    breakout_high[i] = R1_val
                    if abs(close_val - pivot_val) / pivot_val < 0.005:
                        signals[i] = 0.25
                        position = 1
            
            # Exit conditions
            if position == 1:
                # Exit if price reaches S1 or reverses below pivot with weakness
                if (close_val <= S1_val or 
                    (close_val < pivot_val and prices['close'].iloc[i-1] >= pivot_val)):
                    signals[i] = 0.0
                    position = 0
                    breakout_high[i] = np.nan
        
        elif position == -1:
            # Short: waiting for pullback to pivot after S1 breakdown
            if not np.isnan(breakout_low[i-1]):
                # Still waiting for pullback
                if abs(close_val - pivot_val) / pivot_val < 0.01:  # Within 1% of pivot
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
                    position = 0
                    breakout_low[i] = np.nan
            else:
                # No active breakdown, look for new setup
                signals[i] = 0.0
                if (close_val < S1_val and 
                    vol_confirm and 
                    close_val < ema34_val):
                    breakout_low[i] = S1_val
                    if abs(close_val - pivot_val) / pivot_val < 0.005:
                        signals[i] = -0.25
                        position = -1
            
            # Exit conditions
            if position == -1:
                # Exit if price reaches R1 or reverses above pivot with strength
                if (close_val >= R1_val or 
                    (close_val > pivot_val and prices['close'].iloc[i-1] <= pivot_val)):
                    signals[i] = 0.0
                    position = 0
                    breakout_low[i] = np.nan
    
    return signals