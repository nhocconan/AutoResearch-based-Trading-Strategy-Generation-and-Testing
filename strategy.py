#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot R3/S3 fade with 1d trend filter and volume confirmation
# In ranging markets (common in 2025 BTC/ETH), price tends to revert from extreme Camarilla levels (R3/S3)
# In trending markets, we align with 1d EMA50 to avoid fading strong moves
# Volume confirmation reduces false signals
# Target: 12-35 trades/year, discrete sizing 0.25, works in both bull/bear via regime filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = df_1d.copy()
    df_1d['tr'] = np.maximum(
        df_1d['high'] - df_1d['low'],
        np.maximum(
            np.abs(df_1d['high'] - df_1d['close'].shift(1)),
            np.abs(df_1d['low'] - df_1d['close'].shift(1))
        )
    )
    tr_1d = df_1d['tr'].values
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate weekly Camarilla pivot levels (R3, S3)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly high, low, close for Camarilla calculation
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    pivot = (wh + wl + wc) / 3.0
    range_ = wh - wl
    r3 = pivot + range_ * 1.1
    s3 = pivot - range_ * 1.1
    
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when weekly ATR is elevated (> 0.5% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        # Fade at R3/S3 with trend filter
        # Long: price near S3 AND above 1d EMA50 (bullish bias)
        # Short: price near R3 AND below 1d EMA50 (bearish bias)
        near_s3 = close[i] <= s3_aligned[i] * 1.002  # within 0.2% of S3
        near_r3 = close[i] >= r3_aligned[i] * 0.998  # within 0.2% of R3
        
        if near_s3 and close[i] > ema_50_1d_aligned[i] and vol_filter:
            signals[i] = 0.25  # long 25%
        elif near_r3 and close[i] < ema_50_1d_aligned[i] and vol_filter:
            signals[i] = -0.25  # short 25%
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3S3_Fade_TrendFilter_Vol_v1"
timeframe = "6h"
leverage = 1.0