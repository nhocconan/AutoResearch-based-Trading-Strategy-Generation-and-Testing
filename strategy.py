#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1S1_Breakout_Volume_ChopFilter
Hypothesis: In 4h timeframe, trade breakouts of Camarilla R1/S1 levels calculated from daily candles.
Use daily timeframe for pivot calculation to ensure stability. Requires volume confirmation and
choppy market filter (Choppiness Index > 61.8) to avoid false breakouts in low-volatility environments.
Works in bull markets by capturing upside breakouts and in bear markets by capturing downside breakdowns.
Target: 20-40 trades per year with high-conviction entries to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_choppiness_index(high, low, close, window=14):
    """Calculate Choppiness Index: high values indicate ranging market"""
    n = len(high)
    chop = np.full(n, np.nan)
    
    for i in range(window-1, n):
        # True Range
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr = max(tr1, tr2, tr3)
        
        # Sum of True Range over window
        tr_sum = 0
        for j in range(i-window+1, i+1):
            tr1_j = high[j] - low[j]
            tr2_j = abs(high[j] - close[j-1]) if j > 0 else 0
            tr3_j = abs(low[j] - close[j-1]) if j > 0 else 0
            tr_j = max(tr1_j, tr2_j, tr3_j)
            tr_sum += tr_j
        
        # Highest high and lowest low over window
        highest_high = max(high[i-window+1:i+1])
        lowest_low = min(low[i-window+1:i+1])
        
        if tr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * log10(tr_sum / (highest_high - lowest_low)) / log10(window)
        else:
            chop[i] = 50  # neutral
    
    return chop

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    typical = (high + low + close) / 3
    # Camarilla levels
    R4 = close + ((high - low) * 1.5000)
    R3 = close + ((high - low) * 1.2500)
    R2 = close + ((high - low) * 1.1666)
    R1 = close + ((high - low) * 1.0833)
    S1 = close - ((high - low) * 1.0833)
    S2 = close - ((high - low) * 1.1666)
    S3 = close - ((high - low) * 1.2500)
    S4 = close - ((high - low) * 1.5000)
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from daily data
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(
        high_1d, low_1d, close_1d
    )
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed for pivot points)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Calculate Choppiness Index on daily data for regime filter
    chop_1d = calculate_choppiness_index(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only (avoid low-volume Asian session)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Chop filter: only trade in ranging markets (Chop > 61.8) to avoid false breakouts
        chop_ok = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation in ranging market
            if (price > R1_1d_aligned[i] and volume_ok and chop_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation in ranging market
            elif (price < S1_1d_aligned[i] and volume_ok and chop_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches R2 or chop drops below 38.2 (trending market)
            if price >= R2_1d_aligned[i] or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S2 or chop drops below 38.2 (trending market)
            if price <= S2_1d_aligned[i] or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Camarilla_R1S1_Breakout_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0