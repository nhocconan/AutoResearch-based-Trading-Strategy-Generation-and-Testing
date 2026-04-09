#!/usr/bin/env python3
# 6h_camarilla_1d_trend_volume_v3
# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter (EMA50) and volume confirmation.
# Uses tighter volume threshold (2.0x) and requires both trend and price alignment for entry.
# Target: 12-37 trades/year. Works in bull/bear: 1d EMA50 defines trend; Camarilla R4/S4 for breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_1d_trend_volume_v3"
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
    
    # 1d HTF data for EMA trend and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous day's Camarilla pivot levels (using completed 1d bar)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r4_1d = pivot_1d + range_1d * 1.1 / 2
    r3_1d = pivot_1d + range_1d * 1.1 / 4
    s3_1d = pivot_1d - range_1d * 1.1 / 4
    s4_1d = pivot_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average (tighter)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below R3 OR trend turns bearish
            if close[i] < r3_aligned[i] or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 OR trend turns bullish
            if close[i] > s3_aligned[i] or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above R4 with bullish trend
                if close[i] > r4_aligned[i] and close[i] > ema50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below S4 with bearish trend
                elif close[i] < s4_aligned[i] and close[i] < ema50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals