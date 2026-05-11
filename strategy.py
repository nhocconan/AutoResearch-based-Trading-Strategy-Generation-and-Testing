#!/usr/bin/env python3
name = "6h_PivotReversion_Liquidity_Sweep"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivots and liquidity zones
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily high, low, close for pivot calculation
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate daily pivots: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L
    # S2 = P - (H-L), R2 = P + (H-L)
    # S3 = L - 2*(H-P), R3 = H + 2*(P-L) [alternative]
    pivot = (daily_high + daily_low + daily_close) / 3.0
    s1 = 2 * pivot - daily_high
    r1 = 2 * pivot - daily_low
    s2 = pivot - (daily_high - daily_low)
    r2 = pivot + (daily_high - daily_low)
    s3 = daily_low - 2 * (daily_high - pivot)
    r3 = daily_high + 2 * (pivot - daily_low)
    
    # Align pivots to 6h timeframe (wait for daily close)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume filter: 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # ATR for dynamic sizing and exit
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(r1_6h[i]) or 
            np.isnan(s2_6h[i]) or np.isnan(r2_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r3_6h[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false signals
        volume_surge = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long setup: price sweeps below S3 then reverses back above S2
            # This indicates liquidity grab below strong support followed by buying pressure
            if (low[i] < s3_6h[i] and close[i] > s2_6h[i] and volume_surge):
                signals[i] = 0.25
                position = 1
            # Short setup: price sweeps above R3 then reverses back below R2
            # This indicates liquidity grab above strong resistance followed by selling pressure
            elif (high[i] > r3_6h[i] and close[i] < r2_6h[i] and volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            # Dynamic exit: price reaches opposite pivot level or ATR-based target
            if position == 1:
                # Exit long: price reaches R1 (first resistance) or 1.5*ATR adverse move
                if (close[i] >= r1_6h[i]) or (close[i] <= (high[i - 1] if i > 0 else close[i]) - 1.5 * atr[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches S1 (first support) or 1.5*ATR adverse move
                if (close[i] <= s1_6h[i]) or (close[i] >= (low[i - 1] if i > 0 else close[i]) + 1.5 * atr[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals