#!/usr/bin/env python3
"""
1h_camarilla_pivot_4h1d_trend_volume_v1
Hypothesis: Use 4h for trend direction (EMA50) and 1d for Camarilla pivot levels.
Enter on pullbacks to S3/R3 in trending markets with volume confirmation.
Exit at opposite pivot level (S3 for longs, R3 for shorts) or trend reversal.
Timeframe: 1h. Target: 15-35 trades/year per symbol (60-140 over 4 years).
Works in bull/bear by adapting to trend filter (4h EMA50) and using mean reversion at pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_camarilla_pivot_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Daily data for Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2)
    s3 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: 24-period average (1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if data not available
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Trend filter: price relative to 4h EMA50
        above_ema50 = close[i] > ema50_4h_aligned[i]
        below_ema50 = close[i] < ema50_4h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (mean reversion target) or trend turns bearish with volume
            if close[i] <= s3_aligned[i] or (below_ema50 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price reaches R3 (mean reversion target) or trend turns bullish with volume
            if close[i] >= r3_aligned[i] or (above_ema50 and vol_spike):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Only enter in trending markets with volume confirmation
            # In uptrend: buy at S3 pullback
            if above_ema50 and close[i] <= s3_aligned[i] and vol_spike:
                position = 1
                signals[i] = 0.20
            # In downtrend: sell at R3 bounce
            elif below_ema50 and close[i] >= r3_aligned[i] and vol_spike:
                position = -1
                signals[i] = -0.20
    
    return signals