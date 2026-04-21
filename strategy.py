#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1dTrend_VolumeSpike_v1
Hypothesis: Fade extreme Camarilla levels (R3/S3) on 6h with 1d EMA50 trend filter and volume confirmation.
In strong trends (price beyond EMA50), R3/S3 act as exhaustion points where price reverts to the mean (VWAP or EMA20).
In ranging markets, these levels often hold as support/resistance. Volume spike confirms institutional interest at extremes.
Discrete sizing (0.25) and ATR-based trailing exit (1.5x ATR from extreme) minimize fee churn and manage risk.
Target: 12-37 trades/year per symbol for low fee drag and robustness across regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 6h OHLC ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Previous day's Camarilla levels (R3, S3) ===
    cam_high = df_1d['high'].values
    cam_low = df_1d['low'].values
    cam_close = df_1d['close'].values
    
    rng = cam_high - cam_low
    r3 = cam_close + 1.1 * rng  # Camarilla R3
    s3 = cam_close - 1.1 * rng  # Camarilla S3
    
    # Align to 6h (use prior day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(cam_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume filter: current > 2.0x 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14) for trailing exit ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_price = 0.0  # tracks R3/S3 level for trailing exit
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Fade R3: short when price > R3, below EMA50 (exhaustion in uptrend), volume spike
            short_cond = (price > r3_aligned[i]) and (price < ema_50_1d_aligned[i]) and vol_filter
            # Fade S3: long when price < S3, above EMA50 (exhaustion in downtrend), volume spike
            long_cond = (price < s3_aligned[i]) and (price > ema_50_1d_aligned[i]) and vol_filter
            
            if short_cond:
                signals[i] = -0.25
                position = -1
                entry_price = price
                extreme_price = r3_aligned[i]  # trail from R3 level
            elif long_cond:
                signals[i] = 0.25
                position = 1
                entry_price = price
                extreme_price = s3_aligned[i]  # trail from S3 level
        
        elif position == 1:
            # Long: trail exit if price drops 1.5*ATR from extreme (S3 level)
            if price < extreme_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Optional: reverse if price exceeds R3 with volume (strong breakout)
            elif price > r3_aligned[i] and vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: trail exit if price rises 1.5*ATR from extreme (R3 level)
            if price > extreme_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Optional: reverse if price breaks below S3 with volume
            elif price < s3_aligned[i] and vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0