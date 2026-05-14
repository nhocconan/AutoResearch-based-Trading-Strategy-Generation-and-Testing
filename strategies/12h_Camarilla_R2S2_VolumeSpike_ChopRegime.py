#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla Pivot R2/S2 Breakout with Volume Spike and Chop Regime Filter.
Long when price breaks above R2 with volume > 2.0x average in choppy market (CHOP > 61.8).
Short when price breaks below S2 with volume > 2.0x average in choppy market.
Exit when price reverts to pivot point (PP) or chop regime ends (CHOP < 38.2).
Uses 1d for Camarilla pivot calculation, 12h for price/volume, 1d for chop filter.
Target: 50-150 total trades over 4 years (12-37/year). Uses tighter R2/S2 levels and higher volume threshold to reduce trade frequency and improve edge.
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
    
    # Get 1d data for Camarilla pivots and chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R2, S2, PP)
    def calculate_camarilla(high, low, close):
        pp = (high + low + close) / 3.0
        r2 = close + (high - low) * 1.1 / 6.0
        s2 = close - (high - low) * 1.1 / 6.0
        return pp, r2, s2
    
    pp_1d = np.zeros_like(close_1d)
    r2_1d = np.zeros_like(close_1d)
    s2_1d = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        pp, r2, s2 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        pp_1d[i] = pp
        r2_1d[i] = r2
        s2_1d[i] = s2
    
    # Calculate 1d Choppiness Index (CHOP)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(close)
        
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's ATR
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Sum of ATR over period
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Max true range over period
        max_tr = np.zeros_like(close)
        for i in range(period, len(close)):
            max_tr[i] = np.max(tr[i-period+1:i+1])
        
        # Chop formula: 100 * log10(atr_sum / max_tr) / log10(period)
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if max_tr[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / max_tr[i]) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 12h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate volume spike (current volume > 2.0x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r2_1d_aligned[i]) or 
            np.isnan(s2_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        chop_val = chop_1d_aligned[i]
        pp = pp_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        
        # Chop regime: CHOP > 61.8 = ranging (good for mean reversion at pivots)
        is_choppy = chop_val > 61.8
        # Exit chop regime: CHOP < 38.2 = trending (avoid false signals)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long: price breaks above R2 with volume spike in choppy market
            if price > r2 and vol_spike and is_choppy:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume spike in choppy market
            elif price < s2 and vol_spike and is_choppy:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot point OR chop regime ends (trending)
            if price <= pp or is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point OR chop regime ends (trending)
            if price >= pp or is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R2S2_VolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0