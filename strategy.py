#!/usr/bin/env python3
"""
6h_12h_Camarilla_Volume_Signal_v1
Hypothesis: On 6h timeframe, use Camarilla pivot levels from 12h timeframe for entry/exit signals.
Buy near S1/S2 (bullish reversal) with volume confirmation, sell near R1/R2 (bearish reversal) with volume confirmation.
The 12h timeframe provides stable pivot levels that are less noisy than 6h pivots, while volume confirmation
ensures institutional participation. Designed to work in ranging markets where price oscillates around pivots,
which is common in BTC/ETH during accumulation/distribution phases.
Target: 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h data
    # Using previous period's high, low, close
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = high_12h[0]  # First value - no previous
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    
    # Load 6h data for price and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 1:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate 20-period average volume on 6h data
    vol_ma_20 = np.full_like(volume_6h, np.nan)
    for i in range(19, len(volume_6h)):
        vol_ma_20[i] = np.mean(volume_6h[i-19:i+1])
    
    # Align indicators to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 2)  # Volume MA needs 20, pivot needs 1
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        volume_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        volume_ratio = volume_6h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for long entries: price near S1/S2 with volume confirmation
            # Buy when price touches or goes slightly below S1/S2 and reverses up
            if ((close[i] <= s1_aligned[i] * 1.002 or close[i] <= s2_aligned[i] * 1.002) and
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Look for short entries: price near R1/R2 with volume confirmation
            # Sell when price touches or goes slightly above R1/R2 and reverses down
            elif ((close[i] >= r1_aligned[i] * 0.998 or close[i] >= r2_aligned[i] * 0.998) and
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R1 or R2 (take profit) or goes below S2 (stop)
            if (close[i] >= r1_aligned[i] * 0.998 or close[i] >= r2_aligned[i] * 0.998 or
                close[i] <= s2_aligned[i] * 1.002):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches S1 or S2 (take profit) or goes above R2 (stop)
            if (close[i] <= s1_aligned[i] * 1.002 or close[i] <= s2_aligned[i] * 1.002 or
                close[i] >= r2_aligned[i] * 0.998):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_Camarilla_Volume_Signal_v1"
timeframe = "6h"
leverage = 1.0