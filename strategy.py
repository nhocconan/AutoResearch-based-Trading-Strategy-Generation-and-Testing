#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe strategy using weekly pivot points (S3/R3) breakouts with volume confirmation.
# Uses weekly pivots for stronger support/resistance levels, reducing false breakouts.
# Volume surge filter ensures breakouts have conviction. Works in both bull (breakouts continue) and bear (mean reversion at pivots) regimes.
# Target: 20-50 trades over 4 years (5-12/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Points (S3, R3)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r3_1w = pivot_1w + 2 * (high_1w - low_1w)
    s3_1w = pivot_1w - 2 * (high_1w - low_1w)
    
    # Align weekly pivot levels to daily timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Daily ATR for volatility filter (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20  # Require strong volume confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above S3 with volume surge
            if close[i] > s3_1w_aligned[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R3 with volume surge
            elif close[i] < r3_1w_aligned[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to weekly pivot level
            if position == 1:
                if close[i] < pivot_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Pivot_S3_R3_Breakout_1w_Pivot_VolumeSurge_v1"
timeframe = "1d"
leverage = 1.0