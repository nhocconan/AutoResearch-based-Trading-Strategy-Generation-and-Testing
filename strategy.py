#!/usr/bin/env python3
# 6h_AroonOscillator_12hTrend_Volume
# Hypothesis: Uses Aroon Oscillator from 12h timeframe to detect strong trends (values > +50 for uptrend, < -50 for downtrend).
# Entry occurs when Aroon Oscillator crosses above +50 (long) or below -50 (short) with volume confirmation on 6h.
# Exits when Aroon Oscillator returns to neutral zone (-25 to +25) or volume dries up.
# Designed for 6h to capture medium-term trends with reduced whipsaw. Works in both bull and bear via trend-following.

name = "6h_AroonOscillator_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for Aroon Oscillator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Aroon Up and Aroon Down (25-period)
    # Aroon Up = ((25 - periods since highest high) / 25) * 100
    # Aroon Down = ((25 - periods since lowest low) / 25) * 100
    aroon_up = np.full(len(high_12h), np.nan)
    aroon_down = np.full(len(low_12h), np.nan)
    
    for i in range(24, len(high_12h)):
        # Periods since highest high in last 25 periods
        highest_high_idx = np.argmax(high_12h[i-24:i+1])
        periods_since_high = 24 - highest_high_idx
        aroon_up[i] = ((25 - periods_since_high) / 25) * 100
        
        # Periods since lowest low in last 25 periods
        lowest_low_idx = np.argmin(low_12h[i-24:i+1])
        periods_since_low = 24 - lowest_low_idx
        aroon_down[i] = ((25 - periods_since_low) / 25) * 100
    
    # Aroon Oscillator = Aroon Up - Aroon Down
    aroon_osc = aroon_up - aroon_down
    
    # Align Aroon Oscillator to 6h timeframe
    aroon_osc_6h = align_htf_to_ltf(prices, df_12h, aroon_osc)
    
    # Calculate volume spike on 6h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(aroon_osc_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Aroon Oscillator crosses above +50 with volume spike
            if aroon_osc_6h[i] > 50 and aroon_osc_6h[i-1] <= 50 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Aroon Oscillator crosses below -50 with volume spike
            elif aroon_osc_6h[i] < -50 and aroon_osc_6h[i-1] >= -50 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Aroon Oscillator returns to neutral zone (< +25) or volume dries up
            if aroon_osc_6h[i] < 25 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Aroon Oscillator returns to neutral zone (> -25) or volume dries up
            if aroon_osc_6h[i] > -25 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals