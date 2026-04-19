#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_Pivot_Bounce_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Weekly EMA200 for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Camarilla pivot levels from previous day
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12
    r2 = pivot + (prev_high - prev_low) * 1.1 / 6
    s2 = pivot - (prev_high - prev_low) * 1.1 / 6
    r3 = pivot + (prev_high - prev_low) * 1.1 / 4
    s3 = pivot - (prev_high - prev_low) * 1.1 / 4
    r4 = pivot + (prev_high - prev_low) * 1.1 / 2
    s4 = pivot - (prev_high - prev_low) * 1.1 / 2
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200, 14)
    
    for i in range(start_idx, n):
        if np.isnan(atr_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or \
           np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or \
           np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend bias: long bias if price > weekly EMA200, short bias if price < weekly EMA200
        long_bias = price > ema200_1w_aligned[i]
        short_bias = price < ema200_1w_aligned[i]
        
        if position == 0:
            # Long: price touches S1/S2 with volume and bullish bias
            if (abs(price - s1_aligned[i]) < 0.001 * price or abs(price - s2_aligned[i]) < 0.001 * price) and \
               volume_ok and long_bias:
                signals[i] = 0.25
                position = 1
            # Short: price touches R1/R2 with volume and bearish bias
            elif (abs(price - r1_aligned[i]) < 0.001 * price or abs(price - r2_aligned[i]) < 0.001 * price) and \
                 volume_ok and short_bias:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price reaches pivot or R1, or volume dries up
            if price > pivot_aligned[i] * 0.999 or price > r1_aligned[i] * 0.999 or vol < vol_ma * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches pivot or S1, or volume dries up
            if price < pivot_aligned[i] * 1.001 or price < s1_aligned[i] * 1.001 or vol < vol_ma * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals