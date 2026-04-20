#!/usr/bin/env python3
"""
6h_1d_Pivot_R3S3_Fade_v1
Concept: Fade at extreme Camarilla levels (R3/S3) on 1d with mean reversion, confirmed by 6h momentum exhaustion.
- Long: Price <= S3(1d) AND RSI(6) < 30 (oversold)
- Short: Price >= R3(1d) AND RSI(6) > 70 (overbought)
- Exit: Price crosses 1d VWAP (mean reversion target)
- Position sizing: 0.25
- Target: 15-30 trades/year (60-120 total over 4 years)
- Works in bull/bear: Mean reversion at extremes works in all regimes; VWAP exit adapts to trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R3S3_Fade_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivots and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 6h: RSI for momentum exhaustion ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # === Daily: Typical Price for VWAP and Pivots ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    
    # Daily VWAP
    volume_1d = df_1d['volume'].values
    vwap_numerator = np.cumsum(typical_price * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap = vwap_numerator / np.where(vwap_denominator > 0, vwap_denominator, np.nan)
    
    # Daily Camarilla Pivots (based on previous day)
    # Classic formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We use previous day's H,L,C to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    range_1d = prev_high - prev_low
    camarilla_r3 = prev_close + (range_1d * 1.1 / 4)  # R3 level
    camarilla_s3 = prev_close - (range_1d * 1.1 / 4)  # S3 level
    
    # Align daily indicators to 6h
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for RSI and pivots
    
    for i in range(start_idx, n):
        # Get values
        rsi_val = rsi[i]
        close_val = close[i]
        vwap_val = vwap_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(close_val) or np.isnan(vwap_val) or 
            np.isnan(r3_val) or np.isnan(s3_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price at or below S3 AND RSI oversold
            if close_val <= s3_val and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Short: Price at or above R3 AND RSI overbought
            elif close_val >= r3_val and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back above VWAP (mean reversion complete)
            if close_val > vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back below VWAP (mean reversion complete)
            if close_val < vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals