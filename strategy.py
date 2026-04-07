# 6H_CAMARILLA_PIVOT_1D_VOLUME_V1
# Hypothesis: Camarilla pivot levels from daily timeframe act as key support/resistance levels.
# At R3/S3 levels, price tends to reverse (mean reversion) with volume confirmation.
# At R4/S4 levels, price breaks out with continuation (trend following).
# Works in both bull and bear markets as pivots adapt to price action.
# Volume confirmation filters false breakouts/reversals.
# Target: 15-30 trades/year to minimize fee drift on 6h timeframe.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using formulas: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # where C = (H+L+C)/3 (typical price)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Typical price (pivot point)
    pp = (prev_high + prev_low + prev_close) / 3
    # Range
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r4 = pp + range_hl * 1.1 / 2
    r3 = pp + range_hl * 1.1 / 4
    s3 = pp - range_hl * 1.1 / 4
    s4 = pp - range_hl * 1.1 / 2
    
    # Align to 6h timeframe (shifted by 1 day for lookback)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current 6h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (mean reversion target) or closes below S4 (stop)
            if close[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_aligned[i]:  # Stop loss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches R3 (mean reversion target) or closes above R4 (stop)
            if close[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_aligned[i]:  # Stop loss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long at S3 with volume confirmation (bounce from support)
            if close[i] <= s3_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short at R3 with volume confirmation (rejection at resistance)
            elif close[i] >= r3_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
            # Enter long breakout above R4 with volume confirmation
            elif close[i] > r4_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short breakdown below S4 with volume confirmation
            elif close[i] < s4_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals