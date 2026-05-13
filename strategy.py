#!/usr/bin/env python3
"""
6h_Pivot_Volume_Reversal
Hypothesis: In both bull and bear markets, price often reverses at daily Camarilla pivot levels (S3/R3) with volume exhaustion. 
We fade extreme touches of S3/R3 when volume is below average, expecting mean reversion. 
Trades only when price is near the 1-day VWAP to avoid strong trends. 
Designed for low frequency (15-25 trades/year) with clear reversal logic.
"""

name = "6h_Pivot_Volume_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day VWAP for trend filter
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = vwap_num / vwap_den.replace(0, np.nan)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # Using formula: 
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # R2 = C + ((H-L)*1.1/6)
    # R1 = C + ((H-L)*1.1/12)
    # PP = (H+L+C)/3
    # S1 = C - ((H-L)*1.1/12)
    # S2 = C - ((H-L)*1.1/6)
    # S3 = C - ((H-L)*1.1/4)
    # S4 = C - ((H-L)*1.1/2)
    # where C, H, L are from previous day
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla S3 and R3 levels
    rang = prev_high - prev_low
    r3 = prev_close + (rang * 1.1 / 4)
    s3 = prev_close - (rang * 1.1 / 4)
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: below average suggests exhaustion
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_exhausted = volume < (0.7 * vol_ma)  # Volume below 70% of average
    
    # Price near VWAP filter: avoid strong trends
    vwap_diff_pct = np.abs((close - vwap) / vwap)
    near_vwap = vwap_diff_pct < 0.015  # Within 1.5% of VWAP
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if VWAP or Camarilla levels not available
        if np.isnan(vwap[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        # LONG: Price touches or goes below S3 with volume exhaustion, near VWAP
        if low[i] <= s3_aligned[i] and volume_exhausted[i] and near_vwap[i]:
            # Additional confirmation: price closing back above S3 suggests reversal
            if close[i] > s3_aligned[i]:
                signals[i] = 0.25
        # SHORT: Price touches or goes above R3 with volume exhaustion, near VWAP
        elif high[i] >= r3_aligned[i] and volume_exhausted[i] and near_vwap[i]:
            # Additional confirmation: price closing back below R3 suggests reversal
            if close[i] < r3_aligned[i]:
                signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals