#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot breakout with 1-day volume regime filter
# Long when price breaks above Camarilla R4 (1d) AND 1d volume > 1.5x 20-day average
# Short when price breaks below Camarilla S4 (1d) AND 1d volume > 1.5x 20-day average
# Exit when price retouches the Camarilla pivot point (1d)
# Camarilla levels from daily OHLC provide institutional support/resistance
# Volume regime ensures breakouts occur with institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels and volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-day average volume for regime filter (daily)
    vol_1d = df_1d['volume'].values
    vol_avg_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20d)
    
    # Calculate Camarilla levels from daily OHLC (previous day's values)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: 
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # Pivot = (High + Low + Close) / 3
    camarilla_high = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_low = close_1d - 1.5 * (high_1d - low_1d)
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align to 6h timeframe (already delayed by align_htf_to_ltf for completed bar)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_avg_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        vol_threshold = vol_avg_20d_aligned[i]
        
        if position == 0:
            # Long setup: break above Camarilla H4 with volume expansion
            if (price > camarilla_high_aligned[i] and vol_1d_current > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: break below Camarilla L4 with volume expansion
            elif (price < camarilla_low_aligned[i] and vol_1d_current > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Camarilla pivot
            if price <= camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Camarilla pivot
            if price >= camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Camarilla_VolumeRegime_Breakout"
timeframe = "6h"
leverage = 1.0