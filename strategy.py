#!/usr/bin/env python3
# 12h_camarilla_1w_trend_volume_v1
# Hypothesis: 12h Camarilla pivot breakout with 1w EMA50 trend filter and volume confirmation.
# Works in bull/bear: 1w EMA50 defines institutional trend; Camarilla R3/S3/R4/S4 levels from 1d provide
# precise entry/exit levels; volume confirms institutional participation. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivot calculation (using completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for pivot calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = Pivot + Range * 1.1/2
    # R3 = Pivot + Range * 1.1/4
    # S3 = Pivot - Range * 1.1/4
    # S4 = Pivot - Range * 1.1/2
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r4_1d = pivot_1d + range_1d * 1.1 / 2
    r3_1d = pivot_1d + range_1d * 1.1 / 4
    s3_1d = pivot_1d - range_1d * 1.1 / 4
    s4_1d = pivot_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below R3 OR trend turns bearish
            if close[i] < r3_aligned[i] or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above S3 OR trend turns bullish
            if close[i] > s3_aligned[i] or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price breaks above R4 with bullish trend
                if close[i] > r4_aligned[i] and close[i] > ema50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below S4 with bearish trend
                elif close[i] < s4_aligned[i] and close[i] < ema50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals