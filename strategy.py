#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter with 1d Donchian(20) breakout and volume confirmation.
# Uses 12h timeframe to reduce trade frequency. Choppiness Index filters range vs trending markets.
# In trending markets (CHOP < 38.2): breakout entries. In ranging markets (CHOP > 61.8): mean reversion at Donchian bands.
# Daily Donchian channels provide structural support/resistance. Volume confirmation avoids false breakouts.
# Designed to work in both bull (trend follow) and bear (mean reversion in ranges) markets.
name = "12h_Chop_Donchian20_Breakout_1dVol"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Donchian channels and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Donchian(20) channels
    def roll_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def roll_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    high_20 = roll_max(high_1d, 20)
    low_20 = roll_min(low_1d, 20)
    
    # Use previous day's levels to avoid look-ahead
    high_20_prev = np.roll(high_20, 1)
    low_20_prev = np.roll(low_20, 1)
    high_20_prev[0] = np.nan
    low_20_prev[0] = np.nan
    
    # Align to 12h timeframe
    high_20_12h = align_htf_to_ltf(prices, df_1d, high_20_prev)
    low_20_12h = align_htf_to_ltf(prices, df_1d, low_20_prev)
    
    # Daily Choppiness Index (14)
    atr_14 = np.zeros_like(close_1d)
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # first TR
    for i in range(1, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14 if not np.isnan(atr_14[i-1]) else np.nan
    
    # Smoothed ATR for CHOP denominator
    atr_sum = np.zeros_like(close_1d)
    for i in range(len(atr_14)):
        if i < 13:
            atr_sum[i] = np.nan
        else:
            atr_sum[i] = np.nansum(atr_14[i-13:i+1]) if not np.isnan(np.nansum(atr_14[i-13:i+1])) else np.nan
    
    # High-Low range over 14 periods
    hh_14 = roll_max(high_1d, 14)
    ll_14 = roll_min(low_1d, 14)
    range_14 = hh_14 - ll_14
    
    # Choppiness Index: 100 * log10(atr_sum / range_14) / log10(14)
    chop = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(atr_sum[i]) and not np.isnan(range_14[i]) and range_14[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_14[i]) / np.log10(14)
        else:
            chop[i] = np.nan
    
    # Use previous day's CHOP to avoid look-ahead
    chop_prev = np.roll(chop, 1)
    chop_prev[0] = np.nan
    
    # Align to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_prev)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(high_20_12h[i]) or np.isnan(low_20_12h[i]) or
            np.isnan(chop_12h[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Trending market: CHOP < 38.2 -> breakout entries
            if chop_12h[i] < 38.2:
                # Long: break above upper Donchian with volume spike
                if price > high_20_12h[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below lower Donchian with volume spike
                elif price < low_20_12h[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: CHOP > 61.8 -> mean reversion at Donchian bands
            elif chop_12h[i] > 61.8:
                # Long: bounce off lower Donchian (support)
                if price <= low_20_12h[i] * 1.005:  # small tolerance for touch
                    signals[i] = 0.25
                    position = 1
                # Short: bounce off upper Donchian (resistance)
                elif price >= high_20_12h[i] * 0.995:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price reaches opposite Donchian band or CHOP signals range
            if price >= high_20_12h[i] or chop_12h[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches opposite Donchian band or CHOP signals range
            if price <= low_20_12h[i] or chop_12h[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals