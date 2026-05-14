#!/usr/bin/env python3
# 4h_ThreeBarReversal_VolumeFilter_Trend
# Hypothesis: Three-bar reversal patterns (bullish/bearish) at key levels with volume confirmation and daily trend filter work in both bull and bear markets by capturing institutional buying/selling pressure.
# Timeframe: 4h, uses 1d trend filter for multi-timeframe alignment.
# Low trade frequency (~20-30/year) via strict three-bar pattern + volume + trend confluence.
# Long: Bullish reversal (higher low, higher close) above EMA20 with volume > 1.5x average and daily uptrend.
# Short: Bearish reversal (lower high, lower close) below EMA20 with volume > 1.5x average and daily downtrend.
# Exit: Opposite reversal signal or trend failure.
# Uses volume filter to reduce false signals and trend filter for higher timeframe alignment.

timeframe = "4h"
name = "4h_ThreeBarReversal_VolumeFilter_Trend"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA20 for dynamic support/resistance
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Average volume for spike detection (24-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Three-bar reversal detection
    bullish_reversal = np.zeros(n, dtype=bool)
    bearish_reversal = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        # Bullish reversal: higher low and higher close than previous bar
        if low[i] > low[i-1] and close[i] > close[i-1]:
            bullish_reversal[i] = True
        # Bearish reversal: lower high and lower close than previous bar
        if high[i] < high[i-1] and close[i] < close[i-1]:
            bearish_reversal[i] = True
    
    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)  # Ensure we have volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema20[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish reversal above EMA20 with volume spike and daily uptrend
            if bullish_reversal[i] and close[i] > ema20[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish reversal below EMA20 with volume spike and daily downtrend
            elif bearish_reversal[i] and close[i] < ema20[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish reversal or trend failure
            if bearish_reversal[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish reversal or trend failure
            if bullish_reversal[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals