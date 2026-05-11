#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Uses Camarilla pivot levels (R1, S1) from 1-day timeframe for breakout entries on 4h timeframe, filtered by daily trend structure and volume spikes. The Camarilla levels provide natural support/resistance zones where price often reverses or breaks. Combined with daily trend (HH/HL for uptrend, LH/LL for downtrend) and volume confirmation (>1.5x 20-period average), this strategy aims to capture high-probability breakouts in both bull and bear markets. The 4h timeframe reduces trade frequency to avoid excessive fee drag while still capturing significant moves.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and trend structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla levels (R1, S1) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    r1 = pivot + 1.1 * (high_1d - low_1d) / 12.0
    s1 = pivot - 1.1 * (high_1d - low_1d) / 12.0
    
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
    
    # Align 1d indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Camarilla (needs 2 periods) and volume MA(20)
    start_idx = max(2, 20)  # Camarilla needs 2, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
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
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: daily uptrend + volume spike + close above R1 (bullish breakout)
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: daily downtrend + volume spike + close below S1 (bearish breakout)
                if close[i] < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price falls below S1 OR daily uptrend breaks
                if close[i] < s1_aligned[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R1 OR daily downtrend breaks
                if close[i] > r1_aligned[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals