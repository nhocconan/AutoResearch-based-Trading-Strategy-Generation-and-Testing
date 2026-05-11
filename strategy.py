#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Uses daily Camarilla pivot levels (R3/S3) for breakout entries on 4h timeframe, filtered by daily trend structure (HH/HL/LH/LL) and volume spikes.
# Long when: daily uptrend (HH & HL), volume > 1.5x 20-period average, and price breaks above R3 level (bullish breakout).
# Short when: daily downtrend (LH & LL), volume > 1.5x 20-period average, and price breaks below S3 level (bearish breakout).
# Exit when price retouches the central pivot (P) level or daily trend reverses.
# Designed to capture strong breakouts in trending markets while avoiding false signals in low-volume conditions.
# Works in bull markets by catching upward breakouts and in bear markets by catching downward breakdowns.
# Camarilla levels provide statistically significant support/resistance with institutional relevance.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot levels and trend structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla pivot levels (based on previous day) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # P = (H+L+C)/3 (central pivot)
    rng = high_1d - low_1d
    r3 = close_1d + rng * 1.1 / 2
    s3 = close_1d - rng * 1.1 / 2
    p = (high_1d + low_1d + close_1d) / 3
    
    # --- 1d trend structure: HH/HL for uptrend, LH/LL for downtrend ---
    hh = high_1d > np.roll(high_1d, 1)
    hl = low_1d > np.roll(low_1d, 1)
    lh = high_1d < np.roll(high_1d, 1)
    ll = low_1d < np.roll(low_1d, 1)
    uptrend = hh & hl
    downtrend = lh & ll
    uptrend[0] = False
    downtrend[0] = False
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    p_aligned = align_htf_to_ltf(prices, df_1d, p)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Camarilla (needs 1 day) and volume MA(20)
    start_idx = max(1, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(p_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 1d
        is_uptrend = uptrend_aligned[i]
        is_downtrend = downtrend_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: daily uptrend + volume spike + price breaks above R3
                if close[i] > r3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: daily downtrend + volume spike + price breaks below S3
                if close[i] < s3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price retouches central pivot P OR daily uptrend breaks
                if close[i] <= p_aligned[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price retouches central pivot P OR daily downtrend breaks
                if close[i] >= p_aligned[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals