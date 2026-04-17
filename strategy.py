#!/usr/bin/env python3
"""
6h_ElderRay_ForceIndex_Combo
Hypothesis: Elder Ray (Bull/Bear Power) identifies trend strength via EMA13 deviation, 
while Force Index (FI) confirms momentum with volume. Combined, they filter false 
breakouts in chop and capture sustained moves. Works in bull via upward FI divergence, 
in bear via downward FI divergence. Uses 1w trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Force Index (1-period)
    price_change = np.diff(close, prepend=close[0])
    fi_raw = price_change * volume
    fi_smooth = pd.Series(fi_raw).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA40 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema40_1w = close_series_1w.ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Align 1w EMA to 6h timeframe
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(13, 13)  # EMA13, FI smooth
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(fi_smooth[i]) or 
            np.isnan(ema40_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray signals: bull power > 0 and bear power < 0 indicates trend
        # But we use zero-crossing for early signals
        bull_power_cross_up = bull_power[i] > 0 and bull_power[i-1] <= 0
        bear_power_cross_down = bear_power[i] < 0 and bear_power[i-1] >= 0
        
        # Force Index signals: rising FI confirms bullish momentum, falling FI confirms bearish
        fi_rising = fi_smooth[i] > fi_smooth[i-1]
        fi_falling = fi_smooth[i] < fi_smooth[i-1]
        
        if position == 0:
            # Long: bull power crosses above zero + FI rising + 1w uptrend
            if bull_power_cross_up and fi_rising and close[i] > ema40_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bear power crosses below zero + FI falling + 1w downtrend
            elif bear_power_cross_down and fi_falling and close[i] < ema40_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bear power crosses above zero OR FI turns negative
            if bear_power[i] > 0 or fi_smooth[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bull power crosses below zero OR FI turns positive
            if bull_power[i] < 0 or fi_smooth[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ForceIndex_Combo"
timeframe = "6h"
leverage = 1.0