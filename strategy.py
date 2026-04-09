#!/usr/bin/env python3
# 6h_1w_donchian_breakout_v1
# Hypothesis: 6-hour Donchian channel breakouts aligned with weekly trend direction, filtered by volume and volatility.
# Uses weekly Donchian breakout (20-period) as trend filter and 6-hour Donchian breakout (20-period) for entry.
# Only take longs when weekly trend is up (price > weekly Donchian mid) and shorts when weekly trend is down.
# Includes volume confirmation (>1.5x 20-period average) and volatility filter (ATR < 3.5% of price).
# Exit when price returns to the 6-hour Donchian mid-point or when weekly trend flips.
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.full(n, np.nan)
    if n >= 20:
        atr[19] = np.mean(tr[:20])
        for i in range(20, n):
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian high and low (20-period)
    donch_high_1w = np.full(len(df_1w), np.nan)
    donch_low_1w = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i >= 19:
            donch_high_1w[i] = np.max(high_1w[i-19:i+1])
            donch_low_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Weekly Donchian mid-point
    donch_mid_1w = (donch_high_1w + donch_low_1w) / 2.0
    
    # Align weekly Donchian levels to 6h timeframe
    donch_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    donch_mid_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_mid_1w)
    
    # Calculate 6-hour Donchian channel (20-period) for entry signals
    donch_high_6h = np.full(n, np.nan)
    donch_low_6h = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:
            donch_high_6h[i] = np.max(high[i-19:i+1])
            donch_low_6h[i] = np.min(low[i-19:i+1])
    
    # 6-hour Donchian mid-point
    donch_mid_6h = (donch_high_6h + donch_low_6h) / 2.0
    
    # Volume confirmation - 20 period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high_1w_aligned[i]) or np.isnan(donch_low_1w_aligned[i]) or 
            np.isnan(donch_mid_1w_aligned[i]) or np.isnan(donch_high_6h[i]) or 
            np.isnan(donch_low_6h[i]) or np.isnan(donch_mid_6h[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.035 * close[i]  # ATR less than 3.5% of price
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price returns to or below 6h Donchian mid OR weekly trend turns down
            if close[i] <= donch_mid_6h[i] or close[i] < donch_mid_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above 6h Donchian mid OR weekly trend turns up
            if close[i] >= donch_mid_6h[i] or close[i] > donch_mid_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 6h Donchian high with volume confirmation, volatility filter, and weekly uptrend
            if (close[i] > donch_high_6h[i] and vol_ok and vol_filter and 
                close[i] > donch_mid_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 6h Donchian low with volume confirmation, volatility filter, and weekly downtrend
            elif (close[i] < donch_low_6h[i] and vol_ok and vol_filter and 
                  close[i] < donch_mid_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals