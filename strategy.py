#!/usr/bin/env python3
# 1d_weekly_keltner_volume_v1
# Hypothesis: Uses weekly Keltner channels with daily trend filter and volume confirmation.
# Enters long when price breaks above upper Keltner channel with volume spike and daily uptrend.
# Enters short when price breaks below lower Keltner channel with volume spike and daily downtrend.
# Exits on opposite break or trend failure. Designed for 20-30 trades/year to avoid fee drag.
# Uses daily trend filter for multi-timeframe alignment and weekly Keltner channels as structure.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_keltner_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Keltner channels and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for Keltner center
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly ATR(10) for Keltner width
    tr1 = np.maximum(high_1w[1:], close_1w[:-1]) - np.minimum(low_1w[1:], close_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr10_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Keltner channels
    upper_keltner = ema20_1w + 2.0 * atr10_1w
    lower_keltner = ema20_1w - 2.0 * atr10_1w
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to daily timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Align daily data to daily timeframe (identity but keeps pattern)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily volume average (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: breakdown below lower Keltner or daily trend failure
            if close[i] < lower_keltner_aligned[i] or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: break above upper Keltner or daily trend failure
            if close[i] > upper_keltner_aligned[i] or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: above upper Keltner with volume spike and daily uptrend
                if close[i] > upper_keltner_aligned[i] and daily_uptrend:
                    position = 1
                    signals[i] = 0.25
                # Short entry: below lower Keltner with volume spike and daily downtrend
                elif close[i] < lower_keltner_aligned[i] and daily_downtrend:
                    position = -1
                    signals[i] = -0.25
    
    return signals