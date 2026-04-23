#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 level AND close > 1d EMA50 AND volume > 1.8x 24-period average.
Short when price breaks below Camarilla S3 level AND close < 1d EMA50 AND volume > 1.8x 24-period average.
Exit when price crosses Camarilla pivot point (PP) or hits R4/S4 for continuation.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
Camarilla pivots provide precise intraday support/resistance levels that work well on 6h timeframe.
1d EMA50 offers smooth trend filter for alignment with lower lag than slower MA.
Volume confirmation ensures only significant breakouts are taken.
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
    
    # Load 1d data for EMA50 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla pivots from 1d OHLC - ONCE before loop
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    rang = high_1d - low_1d
    
    # Camarilla levels
    r4 = pp + rang * 1.1 / 2
    r3 = pp + rang * 1.1 / 4
    r2 = pp + rang * 1.1 / 6
    r1 = pp + rang * 1.1 / 12
    s1 = pp - rang * 1.1 / 12
    s2 = pp - rang * 1.1 / 6
    s3 = pp - rang * 1.1 / 4
    s4 = pp - rang * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume average (24-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 24)  # Ensure warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND close > 1d EMA50 AND volume spike
            if (price > r3_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND close < 1d EMA50 AND volume spike
            elif (price < s3_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Camarilla pivot point (PP)
            if position == 1 and price < pp_aligned[i]:
                exit_signal = True
            elif position == -1 and price > pp_aligned[i]:
                exit_signal = True
            
            # Alternative exit: continuation breakout at R4/S4 (let winners run)
            elif position == 1 and price > r4_aligned[i]:
                # Continue the trend, but reduce position slightly to lock in profits
                signals[i] = 0.15  # reduce position
                continue  # don't exit, just reduce
            elif position == -1 and price < s4_aligned[i]:
                # Continue the trend, but reduce position slightly to lock in profits
                signals[i] = -0.15  # reduce position
                continue  # don't exit, just reduce
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0