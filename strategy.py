#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h mean reversion at weekly pivot levels with 1d trend filter and volume confirmation.
# Uses weekly R3/S3 for mean reversion entries and R4/S4 for breakout continuations.
# 1d EMA(50) filters trend direction. Volume spike confirms momentum.
# Designed to work in both bull (buy dips at S3 in uptrend) and bear (sell rallies at R3 in downtrend).
# Target: 15-30 trades/year to avoid fee drag on 6h timeframe.
name = "6h_WeeklyPivot_MeanRev_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    # Standard formula: P = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    ph = df_w['high'].values
    pl = df_w['low'].values
    pc = df_w['close'].values
    
    p = (ph + pl + pc) / 3.0
    r1 = 2 * p - pl
    s1 = 2 * p - ph
    r2 = p + (ph - pl)
    s2 = p - (ph - pl)
    r3 = ph + 2 * (p - pl)
    s3 = pl - 2 * (ph - p)
    r4 = r3 + (ph - pl)
    s4 = s3 - (ph - pl)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5x 30-period EMA (moderate threshold)
    vol_ema30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    vol_confirm = volume > (1.5 * vol_ema30)
    
    # Align weekly pivot levels to 6h timeframe (use previous week's values)
    r3_w = align_htf_to_ltf(prices, df_w, r3)
    s3_w = align_htf_to_ltf(prices, df_w, s3)
    r4_w = align_htf_to_ltf(prices, df_w, r4)
    s4_w = align_htf_to_ltf(prices, df_w, s4)
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 week of data for pivots
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(r3_w[i]) or np.isnan(s3_w[i]) or np.isnan(r4_w[i]) or 
            np.isnan(s4_w[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ema30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price at S3 (mean reversion) in uptrend OR break above R4 (continuation)
            # Uptrend: price > EMA50
            if price > ema_50_1d_aligned[i]:
                # Mean reversion at S3
                if price <= s3_w[i] * 1.005:  # Allow 0.5% tolerance
                    signals[i] = 0.25
                    position = 1
                # Breakout continuation above R4
                elif price >= r4_w[i] * 0.995:  # Allow 0.5% tolerance
                    if vol_confirm[i]:
                        signals[i] = 0.25
                        position = 1
            # Downtrend: price < EMA50
            else:
                # Mean reversion at R3
                if price >= r3_w[i] * 0.995:  # Allow 0.5% tolerance
                    signals[i] = -0.25
                    position = -1
                # Breakdown continuation below S4
                elif price <= s4_w[i] * 1.005:  # Allow 0.5% tolerance
                    if vol_confirm[i]:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Exit long: price reaches R3 (take profit) or breaks below S3 (stop)
            if price >= r3_w[i] * 0.995 or price <= s3_w[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S3 (take profit) or breaks above R3 (stop)
            if price <= s3_w[i] * 1.005 or price >= r3_w[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals