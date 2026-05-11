#!/usr/bin/env python3
# 6h_Structure_Breakout_1dTrend_Volume
# Hypothesis: Uses daily structure (higher highs/lows) as trend filter and 6h Donchian(20) breakouts for entry.
# Long when: 1) daily structure is bullish (HH and HL), 2) price breaks above 6h Donchian(20) high, 3) volume > 1.5x 20-period average.
# Short when: 1) daily structure is bearish (LH and LL), 2) price breaks below 6h Donchian(20) low, 3) volume > 1.5x 20-period average.
# Exits when price returns to the 6h Donchian midpoint or structure breaks.
# Works in bull markets by buying pullbacks in uptrends and in bear markets by selling rallies in downtrends.
# Volume confirmation reduces false breakouts. 6h timeframe limits trades to avoid fee drag.

name = "6h_Structure_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for structure (HH, HL, LH, LL)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d structure: HH/HL for uptrend, LH/LL for downtrend ---
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
    
    # --- 6h Donchian(20) ---
    lookback = 20
    # Donchian high: max of last 20 highs
    donch_high = np.full(n, np.nan)
    # Donchian low: min of last 20 lows
    donch_low = np.full(n, np.nan)
    # Donchian mid: average of high and low
    donch_mid = np.full(n, np.nan)
    for i in range(lookback, n):
        donch_high[i] = np.max(high[i-lookback:i])
        donch_low[i] = np.min(low[i-lookback:i])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    # --- 6h volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align all 1d indicators to 6h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend)
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian(20) and volume MA(20)
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(donch_mid[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Structure from 1d
        is_uptrend = uptrend_aligned[i]
        is_downtrend = downtrend_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_uptrend and vol_spike:
                # Long: daily uptrend + volume spike + price above Donchian high
                if close[i] > donch_high[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_downtrend and vol_spike:
                # Short: daily downtrend + volume spike + price below Donchian low
                if close[i] < donch_low[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price returns to Donchian mid OR structure breaks down
                if close[i] < donch_mid[i] or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to Donchian mid OR structure breaks up
                if close[i] > donch_mid[i] or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals