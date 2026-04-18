#!/usr/bin/env python3
"""
6h Weekly Pivot Reversal with Volume and Trend Filter
Hypothesis: In ranging and weak trending markets, price often reverses at weekly pivot levels (S3/R3).
Combined with volume confirmation and 1d EMA trend filter, this strategy fades extremes in ranging
markets and follows breaks of S4/R4 in trending markets, adapting to both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # Pivot = (H + L + C)/3
    pp = (wk_high + wk_low + wk_close) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * pp - wk_low
    s1 = 2 * pp - wk_high
    # R2 = P + (H - L), S2 = P - (H - L)
    r2 = pp + (wk_high - wk_low)
    s2 = pp - (wk_high - wk_low)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3 = wk_high + 2 * (pp - wk_low)
    s3 = wk_low - 2 * (wk_high - pp)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    r4 = r3 + (wk_high - wk_low)
    s4 = s3 - (wk_high - wk_low)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Get 1d EMA for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema50_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        # Weekly pivot levels at this point
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        
        if position == 0:
            # Long setup: price at or below S3 with volume spike in uptrend
            if price <= s3 and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short setup: price at or above R3 with volume spike in downtrend
            elif price >= r3 and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
            # Breakout continuation: price breaks S4/R4 with volume
            elif price < s4 and vol_ok and price < trend:
                signals[i] = 0.25
                position = 1
            elif price > r4 and vol_ok and price > trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches R1 or trend breaks
            if price >= r1_aligned[i] or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S1 or trend breaks
            if price <= s1_aligned[i] or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Reversal_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0