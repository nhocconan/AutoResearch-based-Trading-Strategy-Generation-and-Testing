#!/usr/bin/env python3
# 12h_Vortex_Trend_1dVolatilityBreakout
# Hypothesis: Vortex indicator identifies trend direction on 12h timeframe, while 1d volatility breakout (ATR-based) provides entry timing. Works in both bull and bear markets because Vortex avoids choppy conditions and volatility breakout captures momentum after consolidation. Uses volatility expansion as a regime filter to avoid false breakouts in low-volatility environments.

name = "12h_Vortex_Trend_1dVolatilityBreakout"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate Vortex indicator on 12h for trend direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Vortex Indicator components
    vm_plus = np.abs(high_12h[1:] - low_12h[:-1])
    vm_minus = np.abs(low_12h[1:] - high_12h[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Smooth over 14 periods
    tr14 = np.full_like(tr, np.nan)
    vm_plus_14 = np.full_like(vm_plus, np.nan)
    vm_minus_14 = np.full_like(vm_minus, np.nan)
    
    if len(tr) >= 14:
        tr14[13] = np.nansum(tr[1:15])  # skip first NaN
        vm_plus_14[13] = np.nansum(vm_plus[1:15])
        vm_minus_14[13] = np.nansum(vm_minus[1:15])
        for i in range(14, len(tr)):
            tr14[i] = tr14[i-1] - tr14[i-1]/14 + tr[i]
            vm_plus_14[i] = vm_plus_14[i-1] - vm_plus_14[i-1]/14 + vm_plus[i]
            vm_minus_14[i] = vm_minus_14[i-1] - vm_minus_14[i-1]/14 + vm_minus[i]
    
    # Vortex lines
    vi_plus = vm_plus_14 / tr14
    vi_minus = vm_minus_14 / tr14
    
    # Align Vortex to 12h timeframe (no additional delay needed as it's trend-following)
    vi_plus_aligned = align_htf_to_ltf(prices, df_12h, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_12h, vi_minus)
    
    # Calculate 1d ATR for volatility breakout
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # ATR(10) on 1d
    atr_10_1d = np.full_like(tr_1d, np.nan)
    if len(tr_1d) >= 10:
        atr_10_1d[9] = np.nanmean(tr_1d[1:11])  # skip first NaN
        for i in range(10, len(tr_1d)):
            atr_10_1d[i] = (atr_10_1d[i-1] * 9 + tr_1d[i]) / 10
    
    # Align ATR to 12h timeframe
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    # Volatility breakout: current 12h price vs 12h close of previous period ± ATR
    # We use the 1d ATR scaled to 12h (approx 2x since 1d = 2x12h)
    atr_scaled = atr_10_1d_aligned * 2.0  # approximate scaling for 12h
    
    # Calculate 12h close price for volatility breakout bands
    close_12h_for_vb = np.full_like(close, np.nan)
    close_12h_values = df_12h['close'].values
    aligned_close_12h = align_htf_to_ltf(prices, df_12h, close_12h_values)
    
    # Volatility breakout conditions
    upper_band = aligned_close_12h + atr_scaled
    lower_band = aligned_close_12h - atr_scaled
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 10)  # Ensure Vortex and ATR are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or
            np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: VI+ > VI- (bullish trend) AND price breaks above upper band
            if (vi_plus_aligned[i] > vi_minus_aligned[i] and 
                close[i] > upper_band[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: VI- > VI+ (bearish trend) AND price breaks below lower band
            elif (vi_minus_aligned[i] > vi_plus_aligned[i] and 
                  close[i] < lower_band[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal (VI- > VI+) OR price returns to midline
            midline = aligned_close_12h[i]
            if (vi_minus_aligned[i] > vi_plus_aligned[i] or 
                close[i] < midline):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal (VI+ > VI-) OR price returns to midline
            midline = aligned_close_12h[i]
            if (vi_plus_aligned[i] > vi_minus_aligned[i] or 
                close[i] > midline):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals