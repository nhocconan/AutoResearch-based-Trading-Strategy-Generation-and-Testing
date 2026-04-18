#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Spike + Choppiness Filter
Breakout: Price breaks Donchian(20) high/low
Volume: 2x 20-period volume average
Choppiness: Filter out when CHOP(14) < 38.2 (strong trend) or > 61.8 (choppy) - only trade in between
Long: Breakout above upper band + volume spike + chop filter
Short: Breakout below lower band + volume spike + chop filter
Exit: Opposite breakout or midline crossover
Designed for 12h timeframe to capture medium-term trends with reduced frequency.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_chop(high, low, close, window=14):
    """Calculate Choppiness Index"""
    atr = []
    for i in range(len(high)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr.append(tr)
    
    atr_sum = []
    for i in range(len(atr)):
        if i < window - 1:
            atr_sum.append(np.nan)
        else:
            atr_sum.append(sum(atr[i-window+1:i+1]))
    
    highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
    
    chop = []
    for i in range(len(close)):
        if (np.isnan(atr_sum[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            highest_high[i] == lowest_low[i]):
            chop.append(np.nan)
        else:
            log_val = np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i]))
            chop_val = 100 * log_val / np.log10(window)
            chop.append(chop_val)
    
    return np.array(chop)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index (14-period)
    chop = calculate_chop(high, low, close, 14)
    
    # Get 1d data for additional filter (optional - can be removed if not needed)
    df_1d = get_htf_data(prices, '1d')
    # Example: could add 1d trend filter here if desired
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # need Donchian and chop calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Breakout above upper band + volume spike + chop filter (38.2 < CHOP < 61.8)
            if (price > donch_high[i] and 
                volume_spike[i] and 
                chop[i] > 38.2 and chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower band + volume spike + chop filter
            elif (price < donch_low[i] and 
                  volume_spike[i] and 
                  chop[i] > 38.2 and chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Breakdown below lower band OR price crosses below midline
            if (price < donch_low[i]) or (price < donch_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Breakout above upper band OR price crosses above midline
            if (price > donch_high[i]) or (price > donch_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0