#!/usr/bin/env python3
# 4h_Donchian20_1dTrend_Volume
# Hypothesis: Uses Donchian channel breakouts (20-period) on 4h timeframe, filtered by daily trend structure and volume spikes.
# Long when: daily uptrend (HH & HL), volume > 1.5x 20-period average, and price breaks above Donchian upper band.
# Short when: daily downtrend (LH & LL), volume > 1.5x 20-period average, and price breaks below Donchian lower band.
# Exit when price crosses back through Donchian midpoint or daily trend breaks.
# Designed to capture strong trends with volume confirmation while avoiding false breakouts in low-volume conditions.
# Works in bull markets by catching uptrends early and in bear markets by catching downtrends.
# Donchian channels provide clear breakout levels with built-in trend-following properties.

name = "4h_Donchian20_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend structure (HH, HL, LH, LL)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d trend structure: HH/HL for uptrend, LH/LL for downtrend ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
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
    
    # --- Donchian Channel (20-period) ---
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d trend indicators to 4h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian(20) and volume MA(20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(middle[i]) or
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
                # Long: daily uptrend + volume spike + price breaks above Donchian upper
                if close[i] > upper[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: daily downtrend + volume spike + price breaks below Donchian lower
                if close[i] < lower[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price falls below Donchian middle OR daily uptrend breaks
                if close[i] < middle[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above Donchian middle OR daily downtrend breaks
                if close[i] > middle[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals