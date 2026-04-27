#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime
Hypothesis: 4h strategy using Camarilla R1/S1 levels from 1d for breakout entries with 1d EMA50 trend filter, volume spike confirmation (>2.0x 20-period average), and chop regime filter (Choppiness Index > 61.8 for mean-reversion in ranges). 
Enter long when price closes above R1 with 1d uptrend (price > EMA50) and volume confirmation in chop regime. 
Enter short when price closes below S1 with 1d downtrend (price < EMA50) and volume confirmation in chop regime. 
Exit on opposite Camarilla level touch (S1/R1) or 1d trend reversal (price crosses EMA50). 
Uses discrete position sizing (0.25) to minimize fee churn. Designed for moderate trade frequency (~30-60/year) with regime filter to avoid whipsaws in strong trends and capture mean-reversion in ranges, working in both bull and bear markets.
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
    
    # Get 1d data for Camarilla levels, EMA trend, and chop regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for Camarilla levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R1/S1 from 1d OHLC (tighter than R3/S3 for more precise entries)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(c_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Choppiness Index regime filter on 1d: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending
    # We use chop regime for mean-reversion: only trade when CHOP > 61.8 (ranging market)
    high_low_diff = np.maximum(h_1d - l_1d, 1e-10)  # avoid division by zero
    atr_14 = pd.Series(high_low_diff).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_h_1d = pd.Series(h_1d).rolling(window=14, min_periods=14).max().values
    min_l_1d = pd.Series(l_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.maximum(max_h_1d - min_l_1d, 1e-10)
    chop = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    chop_regime = chop > 61.8  # ranging market for mean-reversion
    
    # Align 1d indicators to 4h timeframe (completed bars only)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d EMA50 (50), volume avg (20), chop (14+14=28)
    start_idx = max(50, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(chop_regime_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        chop_reg = chop_regime_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1d EMA50 trend filter, volume spike, and chop regime
            # Long: price closes above R1 AND above EMA50 (1d uptrend) in chop regime
            long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf and chop_reg
            # Short: price closes below S1 AND below EMA50 (1d downtrend) in chop regime
            short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf and chop_reg
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches S1 (opposite level) OR 1d EMA50 turns bearish (price below EMA)
            if (close_val < s1_val) or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R1 (opposite level) OR 1d EMA50 turns bullish (price above EMA)
            if (close_val > r1_val) or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0