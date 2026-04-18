#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_12hEMA34_Volume_Filtered
Hypothesis: Uses Camarilla pivot levels from daily timeframe combined with EMA34 trend filter on 12h timeframe.
Enters long when price breaks above R1 level with EMA34 rising and volume confirmation.
Enters short when price breaks below S1 level with EMA34 falling and volume confirmation.
Designed for low-moderate trade frequency (~20-50/year) with strong performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY CAMARILLA PIVOT LEVELS (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivots: P = (H+L+C)/3
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    
    camarilla_pivot = typical_price.values
    camarilla_r1 = typical_price.values + hl_range.values * 1.1 / 12
    camarilla_s1 = typical_price.values - hl_range.values * 1.1 / 12
    
    # Align to 4h timeframe with proper delay (use previous day's close for calculation)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 12H EMA34 TREND FILTER (MTF) ===
    df_12h = get_htf_data(prices, '12h')
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # EMA34 slope for trend direction
    ema34_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ema34_aligned[i]) and not np.isnan(ema34_aligned[i-1]):
            ema34_slope[i] = ema34_aligned[i] - ema34_aligned[i-1]
    
    # === VOLUME CONFIRMATION ===
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema34_aligned[i]) or np.isnan(ema34_slope[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with rising EMA34 and volume spike
            if close[i] > r1_aligned[i] and ema34_slope[i] > 0 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with falling EMA34 and volume spike
            elif close[i] < s1_aligned[i] and ema34_slope[i] < 0 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to pivot level or EMA34 turns down
            if close[i] < pivot_aligned[i] or ema34_slope[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to pivot level or EMA34 turns up
            if close[i] > pivot_aligned[i] or ema34_slope[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Breakout_12hEMA34_Volume_Filtered"
timeframe = "4h"
leverage = 1.0