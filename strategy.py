#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Uses Camarilla pivot levels from daily timeframe for breakout entries on 12h timeframe, filtered by daily trend structure and volume spikes.
# Long when: daily uptrend (HH & HL), volume > 1.5x 20-period average, and price breaks above Camarilla R3 level.
# Short when: daily downtrend (LH & LL), volume > 1.5x 20-period average, and price breaks below Camarilla S3 level.
# Exit when price reverses back to Camarilla R4/S4 levels or daily trend breaks.
# Designed to capture strong intraday moves with institutional levels while avoiding false breakouts in low-volume conditions.
# Works in bull markets by catching uptrend breakouts and in bear markets by catching downtrend breakdowns.
# Camarilla levels provide precise support/resistance with statistical edge, especially on 12h timeframe where false signals are filtered.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla pivot levels (R3, S3, R4, S4) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate range
    rng = high_1d - low_1d
    
    # Camarilla levels
    r4 = pp + (rng * 1.1 / 2)
    r3 = pp + (rng * 1.1 / 4)
    s3 = pp - (rng * 1.1 / 4)
    s4 = pp - (rng * 1.1 / 2)
    
    # --- 1d trend structure: HH/HL for uptrend, LH/LL for downtrend ---
    # Higher High: today's high > yesterday's high
    hh = high_1d > np.roll(high_1d, 1)
    # Higher Low: today's low > yesterday's low
    hl = low_1d > np.roll(low_1d, 1)
    # Lower High: today's high < yesterday's high
    lh = high_1d < np.roll(high_1d, 1)
    # Lower Low: today's low < yesterday's low
    ll = low_1d < np.roll(low_1d, 1)
    # Uptrend: HH and HL
    uptrend = hh & hl
    # Downtrend: LH and LL
    downtrend = lh & ll
    # First bar: no previous day, set to False
    uptrend[0] = False
    downtrend[0] = False
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align all 1d indicators to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4)
    uptrend_12h = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_12h = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Camarilla (needs 2 days) and volume MA(20)
    start_idx = max(2, 20)  # Camarilla needs 2 days, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or
            np.isnan(r4_12h[i]) or np.isnan(s4_12h[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(uptrend_12h[i]) or np.isnan(downtrend_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 1d
        is_uptrend = uptrend_12h[i]
        is_downtrend = downtrend_12h[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: daily uptrend + volume spike + price breaks above R3
                if close[i] > r3_12h[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: daily downtrend + volume spike + price breaks below S3
                if close[i] < s3_12h[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price reverses to R4 OR daily uptrend breaks
                if close[i] >= r4_12h[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reverses to S4 OR daily downtrend breaks
                if close[i] <= s4_12h[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals