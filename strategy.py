#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1d Camarilla pivot levels with volume confirmation and chop regime filter.
# Uses daily Camarilla resistance/support levels (R1, S1) for breakout entries, requiring volume above average
# and choppy market conditions (Choppiness Index > 61.8) to avoid false breakouts in strong trends.
# Works in bull markets by buying R1 breakouts and in bear markets by selling S1 breakdowns.
# Target: 20-50 trades per year (80-200 over 4 years) to minimize fee drag.
name = "4h_1d_Camarilla_Pivot_Volume_Chop_Filter"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index: high values indicate ranging, low values indicate trending."""
    atr = np.abs(high - low)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    range_max_min = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum / range_max_min) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    R1 = typical_price + 1.1 * (high_1d - low_1d) / 12.0
    S1 = typical_price - 1.1 * (high_1d - low_1d) / 12.0
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    # Chop filter: Choppiness Index > 61.8 (ranging market)
    chop = calculate_choppiness(high, low, close, 14)
    chop_filter = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for volume MA and chop
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long when price breaks above R1 with volume and chop confirmation
            if close[i] > R1_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume and chop confirmation
            elif close[i] < S1_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls back below R1 or chop drops (trending)
            if close[i] < R1_aligned[i] or chop[i] <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises back above S1 or chop drops (trending)
            if close[i] > S1_aligned[i] or chop[i] <= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals