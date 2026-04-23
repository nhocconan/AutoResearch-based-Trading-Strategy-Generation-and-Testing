#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND close > 1w EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND close < 1w EMA50 AND volume > 2.0x 20-period average.
Exit when price crosses Camarilla H3/L3 levels (mean reversion zones).
Uses discrete position sizing (0.30) to minimize fee churn. Targets 15-35 trades/year per symbol.
Camarilla levels provide proven intraday support/resistance on BTC/ETH pairs.
1w EMA50 offers smooth trend filter for 12h timeframe alignment with very low lag.
Volume confirmation at 2.0x ensures only institutional-grade breakouts are taken.
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
    
    # Load 1w data for EMA50 - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Load 1d data for Camarilla pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels calculation
    # Pivot point
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range
    rng = high_1d - low_1d
    
    # Camarilla levels
    r3 = pp + rng * 1.1 / 4.0  # Resistance 3
    s3 = pp - rng * 1.1 / 4.0  # Support 3
    r4 = pp + rng * 1.1 / 2.0  # Resistance 4
    s4 = pp - rng * 1.1 / 2.0  # Support 4
    h3 = pp + rng * 1.1 / 6.0  # Resistance 3 (alternative naming)
    l3 = pp - rng * 1.1 / 6.0  # Support 3 (alternative naming)
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND close > 1w EMA50 AND volume spike
            if (price > r3_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Camarilla S3 AND close < 1w EMA50 AND volume spike
            elif (price < s3_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Camarilla H3/L3 (mean reversion zones)
            if position == 1 and price < h3_aligned[i]:
                exit_signal = True
            elif position == -1 and price > l3_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12H_Camarilla_R3S3_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0