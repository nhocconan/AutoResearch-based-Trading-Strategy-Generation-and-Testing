#!/usr/bin/env python3
# 6h_camarilla_1d1w_pivot_v1
# Hypothesis: 6h strategy using daily Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
# with weekly trend filter and volume confirmation. Designed for low trade frequency (target: 50-150 total trades
# over 4 years) to avoid fee drag. Works in bull/bear by using weekly EMA trend filter and Camarilla pivot
# structure for entries/exits. Uses discrete sizing (±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_1d1w_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # R3 = close + Range * 1.1/2
    # S3 = close - Range * 1.1/2
    # R4 = close + Range * 1.1
    # S4 = close - Range * 1.1
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 2.0
    s3_1d = close_1d - range_1d * 1.1 / 2.0
    r4_1d = close_1d + range_1d * 1.1
    s4_1d = close_1d - range_1d * 1.1
    
    # Align 1d Camarilla levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (take profit) or breaks below S3 (stop/reversal)
            if close[i] >= r3_aligned[i] or close[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S3 (take profit) or breaks above R3 (stop/reversal)
            if close[i] <= s3_aligned[i] or close[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Determine weekly trend direction
            weekly_uptrend = close[i] > ema_1w_aligned[i]
            weekly_downtrend = close[i] < ema_1w_aligned[i]
            
            if volume_confirmed[i]:
                # Mean reversion long at S3 in uptrend
                if weekly_uptrend and close[i] <= s3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Mean reversion short at R3 in downtrend
                elif weekly_downtrend and close[i] >= r3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                # Breakout long above R4
                elif close[i] >= r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short below S4
                elif close[i] <= s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals