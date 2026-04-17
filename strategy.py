#!/usr/bin/env python3
"""
Hypothesis: 6h Weekly Camarilla Pivot Breakout with Volume Filter.
Long when price breaks above R1 with volume > 1.5x 20-bar average AND 1d close > 1w EMA50 (bullish weekly trend).
Short when price breaks below S1 with volume > 1.5x 20-bar average AND 1d close < 1w EMA50 (bearish weekly trend).
Exit when price returns to the Camarilla pivot point (PP) or weekly trend reverses.
Uses 1d for Camarilla calculation (based on prior 1d bar) and 1w EMA50 for trend filter.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla provides clear breakout levels,
weekly EMA50 ensures alignment with higher-timeframe trend to avoid counter-trend breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (based on prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar: based on prior day's OHLC
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But standard Camarilla uses: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low),
    # R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    # PP = (high + low + close)/3
    # S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low), etc.
    range_1d = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r1 = close_1d + 0.275 * range_1d
    camarilla_s1 = close_1d - 0.275 * range_1d
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d Camarilla levels and 1w EMA50 to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: 1.5x 20-bar average
    volume_s = pd.Series(volume)
    vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pp = camarilla_pp_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume filter AND weekly uptrend
            if price > r1 and vol_filter and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume filter AND weekly downtrend
            elif price < s1 and vol_filter and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot point OR weekly trend reverses
            if price <= pp or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point OR weekly trend reverses
            if price >= pp or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyCamarilla_R1S1_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0