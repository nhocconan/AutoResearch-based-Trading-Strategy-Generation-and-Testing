#!/usr/bin/env python3
# 1d_Donchian20_1wTrend_Volume
# Hypothesis: Uses daily Donchian breakout for trend-following entries, filtered by weekly trend structure and volume spikes.
# Long when: weekly uptrend (HH & HL), volume > 1.5x 20-period average, and price breaks above 20-day high.
# Short when: weekly downtrend (LH & LL), volume > 1.5x 20-period average, and price breaks below 20-day low.
# Exit when price breaks opposite Donchian band or weekly trend reverses.
# Designed to capture major trends with low trade frequency (<25/year) to minimize fee drag.
# Works in bull markets by catching uptrends early and in bear markets by catching downtrends.
# Weekly trend filter reduces false breakouts during ranging periods.

name = "1d_Donchian20_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for trend structure (HH, HL, LH, LL)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w trend structure: HH/HL for uptrend, LH/LL for downtrend ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    # Higher High: today's high > yesterday's high
    hh = high_1w > np.roll(high_1w, 1)
    # Higher Low: today's low > yesterday's low
    hl = low_1w > np.roll(low_1w, 1)
    # Lower High: today's high < yesterday's high
    lh = high_1w < np.roll(high_1w, 1)
    # Lower Low: today's low < yesterday's low
    ll = low_1w < np.roll(low_1w, 1)
    # Uptrend: HH and HL
    uptrend = hh & hl
    # Downtrend: LH and LL
    downtrend = lh & ll
    # First bar: no previous week, set to False
    uptrend[0] = False
    downtrend[0] = False
    
    # --- Donchian channel (20-day high/low) ---
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # --- Volume confirmation (volume > 20-day average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1w trend indicators to daily timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1w, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1w, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian(20) and volume MA(20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 1w
        is_uptrend = uptrend_aligned[i]
        is_downtrend = downtrend_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: weekly uptrend + volume spike + break above 20-day high
                if close[i] > donch_high[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: weekly downtrend + volume spike + break below 20-day low
                if close[i] < donch_low[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price breaks below 20-day low OR weekly uptrend breaks
                if close[i] < donch_low[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above 20-day high OR weekly downtrend breaks
                if close[i] > donch_high[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals