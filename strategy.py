#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_RegimeFilter
Hypothesis: 4h strategy using Camarilla R3/S3 levels from 1d for breakout entries with 1d EMA34 trend filter and chop regime filter. 
Enter long when price closes above R3 with 1d uptrend (price > EMA34) and chop regime (range-bound) for mean reversion exit precision. 
Enter short when price closes below S3 with 1d downtrend (price < EMA34) and chop regime. 
Exit on opposite Camarilla level touch (S3/R3) or 1d trend reversal (price crosses EMA34). 
Designed for moderate trade frequency (~30-60/year) with discrete position sizing (0.25) to reduce fee drag and improve test generalization.
Uses chop regime to avoid false breakouts in strong trends and capture reversals in ranging markets.
Works in both bull and bear markets by following the 1d trend while using Camarilla levels for precise entries and chop filter for regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for Camarilla levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R3/S3 from 1d OHLC (wider than R1/S1 for fewer false breakouts)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_s3 = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(c_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe (completed bars only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Chop regime filter: use 1d data to calculate chop index (range vs trend)
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest high - lowest low)) / log10(14)
    # We'll use a simplified version: high-low range relative to ATR
    tr_1d = np.maximum(h_1d - l_1d, np.maximum(np.abs(h_1d - np.roll(c_1d, 1)), np.abs(l_1d - np.roll(c_1d, 1))))
    tr_1d[0] = h_1d[0] - l_1d[0]  # first bar
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    hl_range_1d = h_1d - l_1d
    chop_1d = 100 * np.log10(hl_range_1d / atr_1d) / np.log10(14)
    # Chop > 61.8 = ranging (good for mean reversion), Chop < 38.2 = trending
    chop_regime = chop_1d > 50  # Using 50 as threshold for ranging regime
    
    # Align chop regime to 4h timeframe
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d EMA34 (34) and chop calculation (14)
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_34_aligned[i]
        chop_reg = chop_regime_aligned[i] > 0.5  # Convert back to boolean
        
        if position == 0:
            # Look for entry: Camarilla R3/S3 breakout with 1d EMA34 trend filter and chop regime (ranging)
            # Long: price closes above R3 AND above EMA34 (1d uptrend) AND chop regime (ranging)
            long_condition = (close_val > r3_val) and (close_val > ema_val) and chop_reg
            # Short: price closes below S3 AND below EMA34 (1d downtrend) AND chop regime (ranging)
            short_condition = (close_val < s3_val) and (close_val < ema_val) and chop_reg
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches S3 (opposite level) OR 1d EMA34 turns bearish (price below EMA)
            if (close_val < s3_val) or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (opposite level) OR 1d EMA34 turns bullish (price above EMA)
            if (close_val > r3_val) or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_RegimeFilter"
timeframe = "4h"
leverage = 1.0