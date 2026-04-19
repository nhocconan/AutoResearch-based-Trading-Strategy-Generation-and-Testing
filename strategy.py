#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Camarilla_R1S1_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Daily EMA200 for trend filter (bull market filter)
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Weekly Camarilla pivot levels (R1, S1)
    prev_close_1w = np.roll(df_1w['close'].values, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w = np.roll(df_1w['high'].values, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w = np.roll(df_1w['low'].values, 1)
    prev_low_1w[0] = np.nan
    
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    r1_1w = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 12.0
    s1_1w = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 12.0
    
    # Align weekly levels to 4h
    pivot_4h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_4h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_4h = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 4h volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for EMA200 warmup
    
    for i in range(start_idx, n):
        if np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or \
           np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0x average
        volume_spike = vol > 2.0 * vol_ma
        
        # Trend filter: price above/below daily EMA200
        price_above_ema = price > ema200_1d_aligned[i]
        price_below_ema = price < ema200_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike AND price above EMA200
            if price > r1_4h[i] and volume_spike and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume spike AND price below EMA200
            elif price < s1_4h[i] and volume_spike and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below pivot (reversal signal)
            if price < pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above pivot (reversal signal)
            if price > pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals