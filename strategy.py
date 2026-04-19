# 6h_Camarilla_R1_S1_Breakout_Volume
# Hypothesis: On 6h timeframe, price breaking above Camarilla R1 or below S1 with volume confirmation indicates strong momentum continuation. Camarilla levels derived from previous day's range provide intraday support/resistance levels that work well in crypto markets. Volume filter ensures breakouts have conviction. Works in both bull and bear markets by capturing momentum bursts.
# Target: 50-150 total trades over 4 years (12-37/year)
# Size: 0.25

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R1_S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # R4 = Close + ((High - Low) * 1.5)
    # R3 = Close + ((High - Low) * 1.25)
    # R2 = Close + ((High - Low) * 1.166)
    # R1 = Close + ((High - Low) * 1.083)
    # S1 = Close - ((High - Low) * 1.083)
    # S2 = Close - ((High - Low) * 1.166)
    # S3 = Close - ((High - Low) * 1.25)
    # S4 = Close - ((High - Low) * 1.5)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (using previous day's values)
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.083)
    s1 = close_1d - (range_1d * 1.083)
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Volume MA and aligned levels ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long when price breaks above R1 with volume
            if price > r1_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume
            elif price < s1_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to S1 (mean reversion to pivot area)
            if price < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to R1 (mean reversion to pivot area)
            if price > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals