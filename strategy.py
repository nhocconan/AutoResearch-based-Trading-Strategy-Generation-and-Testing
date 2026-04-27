#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Choppiness Index regime filter + 12h Donchian breakout.
# In trending regime (CHOP < 38.2): trade Donchian(20) breakout in trend direction.
# In ranging regime (CHOP > 61.8): mean-revert at Donchian boundaries.
# Volume confirmation required for all entries. Targets 20-40 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for regime and trend filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Choppiness Index (14-period)
    chop_period = 14
    atr_12h = np.zeros(len(close_12h))
    for i in range(1, len(close_12h)):
        tr = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
        atr_12h[i] = tr
    
    # Smoothed ATR (simple moving average)
    atr_ma_12h = np.full(len(close_12h), np.nan)
    for i in range(chop_period - 1, len(close_12h)):
        atr_ma_12h[i] = np.mean(atr_12h[i - chop_period + 1:i + 1])
    
    # Calculate Choppiness Index
    chop = np.full(len(close_12h), np.nan)
    for i in range(chop_period - 1, len(close_12h)):
        sum_atr = np.sum(atr_12h[i - chop_period + 1:i + 1])
        highest_high = np.max(high_12h[i - chop_period + 1:i + 1])
        lowest_low = np.min(low_12h[i - chop_period + 1:i + 1])
        if highest_high != lowest_low and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(chop_period)
    
    # Calculate 12h Donchian channels (20-period) for breakout levels
    donch_period = 20
    donch_high_12h = np.full(len(close_12h), np.nan)
    donch_low_12h = np.full(len(close_12h), np.nan)
    for i in range(donch_period - 1, len(close_12h)):
        donch_high_12h[i] = np.max(high_12h[i - donch_period + 1:i + 1])
        donch_low_12h[i] = np.min(low_12h[i - donch_period + 1:i + 1])
    
    # Align 12h indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # 4h Donchian channels (20-period) for entry/exit
    donch_high_4h = np.full(n, np.nan)
    donch_low_4h = np.full(n, np.nan)
    donch_mid_4h = np.full(n, np.nan)
    for i in range(donch_period - 1, n):
        donch_high_4h[i] = np.max(high[i - donch_period + 1:i + 1])
        donch_low_4h[i] = np.min(low[i - donch_period + 1:i + 1])
        donch_mid_4h[i] = (donch_high_4h[i] + donch_low_4h[i]) / 2
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all indicators
    start_idx = max(chop_period - 1, donch_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(donch_high_4h[i]) or 
            np.isnan(donch_low_4h[i]) or np.isnan(donch_mid_4h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        chop_val = chop_aligned[i]
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        if position == 0:
            if is_trending:
                # Trending regime: trade Donchian breakout
                if price > donch_high_4h[i] and vol_filter:
                    signals[i] = size
                    position = 1
                elif price < donch_low_4h[i] and vol_filter:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging regime: mean revert at Donchian boundaries
                if price <= donch_low_4h[i] and vol_filter:
                    signals[i] = size
                    position = 1
                elif price >= donch_high_4h[i] and vol_filter:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Neutral chop: no trade
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses Donchian midline
            if price < donch_mid_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses Donchian midline
            if price > donch_mid_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_ChopRegime_DonchianBreakout_Volume"
timeframe = "4h"
leverage = 1.0