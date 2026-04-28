#!/usr/bin/env python3
"""
6h_Aroon_Oscillator_Trend_With_1dVolume_Spike
Hypothesis: On 6-hour timeframe, use Aroon Oscillator (25-period) to detect strong trends (|AO|>50) and enter in trend direction only when confirmed by 1d volume spike (volume > 1.5x 20-day average). Exit when Aroon Oscillator weakens (|AO|<25) or reverses sign. Designed for moderate trade frequency (~20-40/year) to capture trending moves while avoiding whipsaws in ranging markets. Works in both bull (strong uptrends) and bear (strong downtrends) markets by following the trend direction with volume confirmation.
"""

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
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Aroon Oscillator (25-period)
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Find highest high and lowest low in the period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        # Find periods since high/low
        periods_since_high = np.where(high[i - period + 1:i + 1] == highest_high)[0][-1]
        periods_since_low = np.where(low[i - period + 1:i + 1] == lowest_low)[0][-1]
        
        aroon_up[i] = ((period - 1 - periods_since_high) / (period - 1)) * 100
        aroon_down[i] = ((period - 1 - periods_since_low) / (period - 1)) * 100
    
    aroon_osc = aroon_up - aroon_down  # Range: -100 to +100
    
    # 1d volume spike: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (vol_ma_20 * 1.5)
    
    # Align volume spike to 6h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = period  # Wait for Aroon calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(aroon_osc[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        ao = aroon_osc[i]
        vol_spike = volume_spike_aligned[i]
        
        # Entry: strong trend (|AO|>50) with volume spike
        long_entry = ao > 50 and vol_spike
        short_entry = ao < -50 and vol_spike
        
        # Exit: trend weakens (|AO|<25) or reverses sign
        long_exit = (ao < 25) or (ao < 0 and position == 1)
        short_exit = (ao > -25) or (ao > 0 and position == -1)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Aroon_Oscillator_Trend_With_1dVolume_Spike"
timeframe = "6h"
leverage = 1.0