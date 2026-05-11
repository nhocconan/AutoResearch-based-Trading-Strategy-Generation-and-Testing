#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Uses daily Camarilla pivot levels (R1/S1) for breakout entries, filtered by daily trend (price vs EMA34) and volume spikes. Designed for low trade frequency (20-50/year) by requiring confluence of three conditions: price breaking R1/S1, aligned with daily trend, and volume confirmation. Works in bull/bear markets by following higher-timeframe trend while using price action for precise entries.
"""

name = "4h_1d_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price for the period
    typical_price = (high + low + close) / 3
    # Range
    range_val = high - low
    
    # Camarilla levels
    pivot = typical_price
    r1 = close + (range_val * 1.1 / 12)
    s1 = close - (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    s2 = close - (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    s3 = close - (range_val * 1.1 / 4)
    r4 = close + (range_val * 1.1 / 2)
    s4 = close - (range_val * 1.1 / 2)
    
    return r1, s1, r2, s2, r3, s3, r4, s4, pivot

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla Pivot Levels (from previous day) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous day
    r1, s1, r2, s2, r3, s3, r4, s4, pivot = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    r2 = np.roll(r2, 1)
    s2 = np.roll(s2, 1)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    r4 = np.roll(r4, 1)
    s4 = np.roll(s4, 1)
    pivot = np.roll(pivot, 1)
    # Set first day's levels to NaN (no previous day)
    r1[0] = np.nan
    s1[0] = np.nan
    r2[0] = np.nan
    s2[0] = np.nan
    r3[0] = np.nan
    s3[0] = np.nan
    r4[0] = np.nan
    s4[0] = np.nan
    pivot[0] = np.nan
    
    # Align daily Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # --- Daily Trend Filter (EMA34) ---
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34)
    
    # --- Volume Spike Detection (20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema34_4h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R1 + above daily EMA34 + volume spike
            if (close[i] > r1_4h[i] and 
                close[i] > ema34_4h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + below daily EMA34 + volume spike
            elif (close[i] < s1_4h[i] and 
                  close[i] < ema34_4h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite break of S1/R1 or volume dissipation
            if position == 1:
                # Exit long: price breaks below S1 OR volume drops below average
                if close[i] < s1_4h[i] or vol_ratio[i] < 1.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R1 OR volume drops below average
                if close[i] > r1_4h[i] or vol_ratio[i] < 1.0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals