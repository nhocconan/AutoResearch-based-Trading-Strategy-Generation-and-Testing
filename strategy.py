#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w pivot direction and 1d ATR-based volatility filter.
# Uses weekly Camarilla pivot levels (R3/S3) for mean reversion entries with 1d ATR filter to avoid low volatility periods.
# Designed to work in both bull and bear markets by fading extremes in ranging conditions.
# Target: 12-37 trades per year to minimize fee drag.
name = "6h_WeeklyCamarilla_R3S3_MeanRev_1dATRFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # R3 = Pivot + 1.1 * (H - L)
    # S3 = Pivot - 1.1 * (H - L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r3 = pivot + 1.1 * (weekly_high - weekly_low)
    s3 = pivot - 1.1 * (weekly_high - weekly_low)
    
    # Align weekly R3/S3 to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get daily data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-day ATR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    atr_14 = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        else:
            atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Align ATR to 6h timeframe
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # ATR 20-period average for volatility regime filter
    atr_ma = pd.Series(atr_14_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Wait for ATR MA and ATR calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(atr_14_6h[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR > 1.5x its 20-period average
        vol_filter = atr_14_6h[i] > 1.5 * atr_ma[i]
        
        if position == 0:
            # Long: price below S3 with volatility expansion (mean reversion long)
            if close[i] < s3_6h[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price above R3 with volatility expansion (mean reversion short)
            elif close[i] > r3_6h[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above pivot or volatility collapses
            if close[i] > pivot[-1 if len(pivot) == 1 else np.sum(~np.isnan(pivot))] or atr_14_6h[i] < atr_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below pivot or volatility collapses
            if close[i] < pivot[-1 if len(pivot) == 1 else np.sum(~np.isnan(pivot))] or atr_14_6h[i] < atr_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals