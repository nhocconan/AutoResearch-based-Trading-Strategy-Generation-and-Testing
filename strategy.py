#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla Pivot reversal with 1-day volume confirmation
# Long when price touches L3/H3 support/resistance AND volume > 1.8x 20-period average
# Short when price touches H3/L3 resistance/support AND volume > 1.8x 20-period average
# Exit when price reaches opposite H4/L4 level or reverses back to H3/L3
# Uses Camarilla pivot levels from daily timeframe for precise reversal zones
# Volume confirmation ensures institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) for low friction

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    H4 = close_1d + range_1d * 1.1 / 2
    H3 = close_1d + range_1d * 1.1 / 4
    L3 = close_1d - range_1d * 1.1 / 4
    L4 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or 
            np.isnan(H4_12h[i]) or np.isnan(L4_12h[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.8
        
        if position == 0:
            # Long setup: price touches L3 support with volume confirmation
            if (abs(price - L3_12h[i]) < 0.001 * price and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price touches H3 resistance with volume confirmation
            elif (abs(price - H3_12h[i]) < 0.001 * price and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H4 or returns to L3
            if price >= H4_12h[i] or abs(price - L3_12h[i]) < 0.001 * price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L4 or returns to H3
            if price <= L4_12h[i] or abs(price - H3_12h[i]) < 0.001 * price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Volume_Reversal"
timeframe = "12h"
leverage = 1.0