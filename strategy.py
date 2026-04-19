#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with 1-day pivot range filter.
# Williams %R identifies overbought/oversold conditions; pivot ranges define institutional support/resistance.
# Long when: %R < -80 (oversold) and price > daily pivot (bullish bias)
# Short when: %R > -20 (overbought) and price < daily pivot (bearish bias)
# Exit when: %R crosses above -50 (for long) or below -50 (for short)
# Volume confirmation: current volume > 1.3x 20-period average to filter weak moves.
# Works in ranging markets (mean reversion at extremes) and trending markets (pullbacks to pivot).
# Target: 15-25 trades/year per symbol.
name = "6h_WilliamsR_Pivot_Range_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate daily pivot points (using 1D data)
    df_1d = get_htf_data(prices, '1d')
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pivot = (typical_price + df_1d['high'] + df_1d['low']) / 3
    pivot_values = pivot.values
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_values)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # Wait for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        price = close[i]
        pivot_val = pivot_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Oversold and price above pivot (bullish bias) with volume confirmation
            if (wr < -80 and price > pivot_val and vol > 1.3 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Overbought and price below pivot (bearish bias) with volume confirmation
            elif (wr > -20 and price < pivot_val and vol > 1.3 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (momentum fading)
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (momentum fading)
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals