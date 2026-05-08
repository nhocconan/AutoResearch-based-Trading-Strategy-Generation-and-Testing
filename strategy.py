#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d CAMARILLA pivot breakouts with volume confirmation.
# In trending markets (CHOP < 38.2), trade breakouts of CAMARILLA H3/L3 levels.
# In ranging markets (CHOP > 61.8), trade mean reversion at H4/L4 levels.
# Volume must exceed 1.5x 20-period average to confirm breakout/mean reversion.
# Uses 4h timeframe for execution, with 1d pivots and 4h chop filter.
# Target: 80-150 total trades over 4 years (20-38/year) to balance edge and fee drag.

name = "4h_Choppiness_Camarilla_Pivot_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for CAMARILLA pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h Choppiness Index (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # True Range for chop denominator
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(TR_sum / (ATR * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Calculate 1d CAMARILLA pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # CAMARILLA formulas
    H5 = close_1d + 1.1 * (high_1d - low_1d) / 2
    H4 = close_1d + 1.1 * (high_1d - low_1d) / 4
    H3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    L3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    L4 = close_1d - 1.1 * (high_1d - low_1d) / 4
    L5 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align CAMARILLA levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for chop and volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine market regime based on Choppiness Index
            if chop[i] < 38.2:  # Trending market
                # Breakout strategy: enter on H3/L3 break with volume
                long_break = (close[i] > H3_aligned[i]) and volume_filter[i]
                short_break = (close[i] < L3_aligned[i]) and volume_filter[i]
                
                if long_break:
                    signals[i] = 0.25
                    position = 1
                elif short_break:
                    signals[i] = -0.25
                    position = -1
            elif chop[i] > 61.8:  # Ranging market
                # Mean reversion strategy: enter at H4/L4 with volume
                long_reversion = (close[i] < L4_aligned[i]) and volume_filter[i]
                short_reversion = (close[i] > H4_aligned[i]) and volume_filter[i]
                
                if long_reversion:
                    signals[i] = 0.20
                    position = 1
                elif short_reversion:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit conditions
            if chop[i] < 38.2:  # Trending: exit on H4 break or L3 retest
                if close[i] > H4_aligned[i] or close[i] < L3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging: exit at H3 or mean reversion
                if close[i] > H3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            # Short exit conditions
            if chop[i] < 38.2:  # Trending: exit on L4 break or H3 retest
                if close[i] < L4_aligned[i] or close[i] > H3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging: exit at L3 or mean reversion
                if close[i] < L3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals