#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakout on 6h with 1-week EMA34 trend filter and volume spike confirmation.
Uses weekly EMA34 for trend direction (captures major trend) with volume spike to confirm institutional participation.
Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag while working in both bull and bear markets.
Weekly trend filter helps avoid counter-trend trades during major moves, while R3/S3 levels provide significant breakout points.
"""

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
    
    # Calculate 1-week EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1-week Camarilla pivot levels (R3, S3)
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    PP = (high_1w + low_1w + close_1w) / 3.0
    R3 = PP + (high_1w - low_1w) * 1.0 / 4.0
    S3 = PP - (high_1w - low_1w) * 1.0 / 4.0
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(34, 20, 14)  # EMA, volume avg, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i]) or
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        ema_trend = ema_1w_aligned[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for breakout in direction of weekly trend with volume confirmation
            # Long: price above weekly EMA34 AND break above R3 + volume spike
            long_entry = (close_val > ema_trend) and (close_val > R3_aligned[i]) and volume_spike[i]
            # Short: price below weekly EMA34 AND break below S3 + volume spike
            short_entry = (close_val < ema_trend) and (close_val < S3_aligned[i]) and volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on S3 retracement or ATR stoploss
            exit_condition = (close_val < S3_aligned[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R3 retracement or ATR stoploss
            exit_condition = (close_val > R3_aligned[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0