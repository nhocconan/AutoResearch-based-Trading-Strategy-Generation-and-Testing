#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dRegime_v1
Hypothesis: On 1h timeframe, trade Camarilla R1/S1 breakouts from prior 1h bar with 4h EMA50 trend filter and 1d chop regime filter. Target 15-37 trades/year by requiring confluence of HTF trend alignment, low-chop regime, and price structure breakout. Designed to work in both bull and bear markets via trend filter and regime avoidance of sideways chop.
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
    
    # Get 4h data for HTF trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1h data for Camarilla levels
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1h bar (HLC of prior 1h)
    cam_high = pd.Series(df_1h['high'].values).shift(1).values
    cam_low = pd.Series(df_1h['low'].values).shift(1).values
    cam_close = pd.Series(df_1h['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels (core breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Choppiness Index (CHOP) on 1d: high CHOP = ranging market (avoid), low CHOP = trending (favor)
    # CHOP = 100 * log10(sum(ATR over n) / (log(n) * (max(high) - min(low)))) / log10(n)
    atr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                        np.maximum(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                                   np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))))
    atr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (np.log(14) * (max_high - min_low))) / np.log10(14)
    
    # Align HTF indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    R1_aligned = align_htf_to_ltf(prices, df_1h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1h, S1)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(50) 4h, Camarilla (need 2 bars for shift), CHOP (14)
    start_idx = max(50, 2, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        chop_val = chop_aligned[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_4h_val
        downtrend = close_val < ema_50_4h_val
        
        # Regime filter: only trade when market is trending (CHOP < 38.2) or moderately choppy (CHOP < 50)
        # Avoid strong ranging markets (CHOP > 61.8)
        regime_filter = chop_val < 50.0
        
        if position == 0:
            # Long: break above R1 with uptrend and favorable regime
            long_signal = (close_val > r1_val) and \
                          uptrend and \
                          regime_filter
            
            # Short: break below S1 with downtrend and favorable regime
            short_signal = (close_val < s1_val) and \
                           downtrend and \
                           regime_filter
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            highest_since_entry = max(highest_since_entry, high_val)
            # Time-based exit: exit after 24 hours (24 bars on 1h)
            # Optional: could add ATR stop or regime change exit
            if chop_val > 61.8:  # Exit if market becomes strongly ranging
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            lowest_since_entry = min(lowest_since_entry, low_val)
            # Time-based exit: exit after 24 hours (24 bars on 1h)
            if chop_val > 61.8:  # Exit if market becomes strongly ranging
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dRegime_v1"
timeframe = "1h"
leverage = 1.0