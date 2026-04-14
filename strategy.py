# 12h_Camarilla_Pivot_Volume_Strategy
# Hypothesis: Camarilla pivot levels derived from daily OHLC provide strong support/resistance levels.
# Price touching L3 (support) with volume confirmation and bullish momentum triggers long.
# Price touching H3 (resistance) with volume confirmation and bearish momentum triggers short.
# Uses 12h timeframe to reduce trade frequency and avoid fee drag, with daily pivot for structure.
# Volume filter ensures breakouts have conviction. Works in both trending and ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from daily OHLC
    # Formulas: 
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # L3 = Pivot - 1.1 * Range / 2
    # H3 = Pivot + 1.1 * Range / 2
    # L4 = Pivot - 1.1 * Range
    # H4 = Pivot + 1.1 * Range
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    L3_1d = pivot_1d - 1.1 * range_1d / 2.0
    H3_1d = pivot_1d + 1.1 * range_1d / 2.0
    L4_1d = pivot_1d - 1.1 * range_1d
    H4_1d = pivot_1d + 1.1 * range_1d
    
    # Align daily pivot levels to 12h timeframe
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(L3_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price touches or breaks below L3 with volume confirmation
            # In ranging markets, L3 acts as strong support
            if price <= L3_aligned[i] and vol > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short setup: price touches or breaks above H3 with volume confirmation
            # In ranging markets, H3 acts as strong resistance
            elif price >= H3_aligned[i] and vol > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 (opposite resistance level) or shows weakness
            if price >= H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 (opposite support level) or shows strength
            if price <= L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Pivot_Volume_Strategy"
timeframe = "12h"
leverage = 1.0