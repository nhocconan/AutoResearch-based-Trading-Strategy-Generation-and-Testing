#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_Filter
Hypothesis: Camarilla R3/S3 breakouts on 12h timeframe with 1w EMA34 trend filter. 
Only trade breakouts in direction of weekly trend to avoid counter-trend whipsaws.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade 
frequency (~12-25/year) to work in both bull and bear markets via HTF trend alignment.
Camarilla levels provide institutional support/resistance with higher reliability 
than standard Donchian breaks in ranging markets. Weekly trend filter ensures we 
only trade with the dominant multi-week momentum.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF EMA34 to 12h timeframe (standard 1-bar delay for EMA)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    
    # Get 1d data for Camarilla pivot calculation (standard practice)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use the previous completed 1d bar to avoid look-ahead
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous 1d bar's OHLC (index i-1 in 1d data corresponds to bar before current 12h bar)
        # Find the 1d bar that completed before current 12h bar
        # Since we're on 12h timeframe, we need to map to daily bars
        # Simpler approach: use the 1d bar from 2 bars ago to ensure it's completed
        if i >= 2:
            prev_close = close_1d[-(i//2 + 1)] if (i//2 + 1) <= len(close_1d) else close_1d[-1]
            prev_high = high_1d[-(i//2 + 1)] if (i//2 + 1) <= len(high_1d) else high_1d[-1]
            prev_low = low_1d[-(i//2 + 1)] if (i//2 + 1) <= len(low_1d) else low_1d[-1]
            
            # More robust: use rolling window of 1d data aligned to 12h
            pass
    
    # Simpler approach: calculate Camarilla on 1d then align
    # Typical Camarilla calculation
    HLDiff = high_1d - low_1d
    camarilla_R3_1d = close_1d + (HLDiff * 1.1 / 4)
    camarilla_S3_1d = close_1d - (HLDiff * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3_1d)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and enough data for alignment
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or
            np.isnan(camarilla_S3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for Camarilla breakout signals with weekly trend filter
            # Long: price breaks above R3 in uptrend (close > EMA34_1w)
            # Short: price breaks below S3 in downtrend (close < EMA34_1w)
            long_signal = (close[i] > camarilla_R3_aligned[i]) and (close[i] > ema34_aligned[i])
            short_signal = (close[i] < camarilla_S3_aligned[i]) and (close[i] < ema34_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below R3 (mean reversion) or trend turns
            exit_signal = (close[i] < camarilla_R3_aligned[i]) or (close[i] < ema34_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above S3 (mean reversion) or trend turns
            exit_signal = (close[i] > camarilla_S3_aligned[i]) or (close[i] > ema34_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Filter"
timeframe = "12h"
leverage = 1.0