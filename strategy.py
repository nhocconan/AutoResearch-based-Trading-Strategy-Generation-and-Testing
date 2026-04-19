#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_MomentumBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # Fix first value
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1w pivot levels for directional bias
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    prev_high_1w = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low_1w = np.concatenate([[np.nan], low_1w[:-1]])
    prev_close_1w = np.concatenate([[np.nan], close_1w[:-1]])
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r1_1w = 2 * pivot_1w - prev_low_1w
    s1_1w = 2 * pivot_1w - prev_high_1w
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 6h momentum: price change over 3 periods (18h)
    mom_6h = (close - np.roll(close, 3)) / np.roll(close, 3)
    mom_6h[:3] = 0  # First 3 values invalid
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 200, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(atr_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or \
           np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or \
           np.isnan(s1_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(mom_6h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        mom = mom_6h[i]
        
        # Volume filter
        volume_ok = vol > 1.3 * vol_ma
        
        # Momentum filter
        mom_ok = abs(mom) > 0.015  # 1.5% momentum threshold
        
        # Trend bias: long bias if price > EMA200, short bias if price < EMA200
        long_bias = price > ema200_1d_aligned[i]
        short_bias = price < ema200_1d_aligned[i]
        
        # Weekly pivot bias: long bias if above weekly pivot, short bias if below
        pw_long_bias = price > pivot_1w_aligned[i]
        pw_short_bias = price < pivot_1w_aligned[i]
        
        if position == 0:
            # Long: bullish momentum + above EMA200 + above weekly pivot + volume
            if mom > 0 and mom_ok and long_bias and pw_long_bias and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum + below EMA200 + below weekly pivot + volume
            elif mom < 0 and mom_ok and short_bias and pw_short_bias and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: momentum turns bearish or price drops below weekly pivot
            if mom < -0.005 or price < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: momentum turns bullish or price rises above weekly pivot
            if mom > 0.005 or price > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals