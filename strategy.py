#!/usr/bin/env python3
"""
6h_1d_Camarilla_R1S1_Breakout_Volume
Hypothesis: Uses Camarilla pivot levels (R1/S1) from daily timeframe with 6h breakout confirmation.
Enters long when price breaks above R1 with volume confirmation, short when breaks below S1 with volume confirmation.
Uses daily trend filter (price above/below 50-period EMA) to align with higher timeframe trend.
Designed for moderate trade frequency (~15-25/year) with trend-following capability in both bull and bear markets.
Camarilla levels work well in ranging markets while breakouts capture trending moves.
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1/12
    # S1 = C - (H - L) * 1.1/12
    # R4 = C + (H - L) * 1.1/2
    # S4 = C - (H - L) * 1.1/2
    
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    
    camarilla_r1 = typical_price + hl_range * 1.1 / 12
    camarilla_s1 = typical_price - hl_range * 1.1 / 12
    camarilla_r4 = typical_price + hl_range * 1.1 / 2
    camarilla_s4 = typical_price - hl_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Daily trend filter: 50-period EMA
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.8x 24-period average (48h)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above daily EMA50
            if close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and vol_spike[i] and close[i] > ema50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and below daily EMA50
            elif close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and vol_spike[i] and close[i] < ema50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S1 or reverses below daily EMA50
            if close[i] < s1_aligned[i] or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R1 or reverses above daily EMA50
            if close[i] > r1_aligned[i] or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Camarilla_R1S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0