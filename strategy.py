#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_WeeklyPivotDirection_VolumeSpike_v1
Hypothesis: 6h Camarilla R1/S1 breakout with weekly pivot direction filter (price above/below weekly pivot) and volume confirmation (>1.8x 20-period MA). Weekly pivot provides longer-term trend bias that works in both bull/bear markets by requiring alignment with institutional levels. Volume spike confirms institutional participation. Target: 80-150 total trades over 4 years (20-38/year) to balance opportunity with fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla, 1w for pivot)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 1:
        return np.zeros(n)
    
    # === 1d Camarilla pivot levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = prev_low_1d[0] = prev_close_1d[0] = np.nan
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    r1_1d = pivot_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    s1_1d = pivot_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1w pivot for trend direction ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w[-1] + low_1w[-1] + close_1w[-1]) / 3.0  # use last completed weekly pivot
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(close_1w, pivot_1w), additional_delay_bars=1)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (1.8x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(pivot_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        pivot_1w_val = pivot_1w_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.8x average (strict threshold)
        volume_confirm = volume_now > 1.8 * vol_avg
        
        if position == 0:
            # Long: price breaks above R1, above weekly pivot, volume confirm
            long_condition = (price > r1_val) and (price > pivot_1w_val) and volume_confirm
            # Short: price breaks below S1, below weekly pivot, volume confirm
            short_condition = (price < s1_val) and (price < pivot_1w_val) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below weekly pivot)
                elif price < pivot_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above weekly pivot)
                elif price > pivot_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_WeeklyPivotDirection_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0