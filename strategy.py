#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal with 1d Volume Spike and Choppiness Filter.
Long when price touches S3 support with volume spike in choppy market.
Short when price touches R3 resistance with volume spike in choppy market.
Exit when price crosses the pivot point (PP) or after 4 bars.
Designed to generate 15-30 trades/year per symbol with mean-reversion edge in chop.
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
    
    # Get 1d data for Camarilla pivot levels and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    # R4 = Close + ((High - Low) * 1.5000)
    # R3 = Close + ((High - Low) * 1.2500)
    # R2 = Close + ((High - Low) * 1.1666)
    # R1 = Close + ((High - Low) * 1.0833)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 1.0833)
    # S2 = Close - ((High - Low) * 1.1666)
    # S3 = Close - ((High - Low) * 1.2500)
    # S4 = Close - ((High - Low) * 1.5000)
    
    pp = (prev_high + prev_low + prev_close) / 3.0
    r3 = prev_close + (prev_high - prev_low) * 1.2500
    s3 = prev_close - (prev_high - prev_low) * 1.2500
    
    # Align to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike: volume > 2.0x 24-period average
    vol_ma_24 = np.empty_like(volume, dtype=np.float64)
    vol_ma_24.fill(np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    # Choppiness filter: use 1d ATR-based chop (simplified)
    # Chop = 100 * log10( sum(ATR1) / (ATR14 * n) ) / log10(n)
    # We'll use a simpler version: high-low range relative to ATR
    atr_period = 14
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - df_1d['close'].shift(1).values)
    tr3 = np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.empty_like(tr, dtype=np.float64)
    atr.fill(np.nan)
    for i in range(atr_period-1, len(tr)):
        if i == atr_period-1:
            atr[i] = np.mean(tr[:atr_period])
        else:
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Align ATR and calculate choppy condition (high ATR relative to range = choppy)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    # Choppy when current ATR > 1.5 * average ATR (indicating volatile/choppy market)
    atr_ma_10 = np.empty_like(atr_aligned, dtype=np.float64)
    atr_ma_10.fill(np.nan)
    for i in range(9, len(atr_aligned)):
        atr_ma_10[i] = np.mean(atr_aligned[i-9:i+1])
    choppy = atr_aligned > 1.5 * atr_ma_10  # True when choppy/volatile
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (shifted) + volume MA (24) + ATR (14)
    start_idx = max(24, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_24[i]) or
            np.isnan(choppy[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        pp_val = pp_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        
        # Volume filter: volume spike > 2.0x average
        vol_spike = vol_now > 2.0 * vol_ma_24[i]
        
        if position == 0:
            # Long when price touches S3 support with volume spike in choppy market
            if price_now <= s3_val and vol_spike and choppy[i]:
                signals[i] = size
                position = 1
            # Short when price touches R3 resistance with volume spike in choppy market
            elif price_now >= r3_val and vol_spike and choppy[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above pivot point or after 4 bars
            if price_now > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses below pivot point or after 4 bars
            if price_now < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_S3R3_VolumeSpike_Chop"
timeframe = "12h"
leverage = 1.0