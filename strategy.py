#!/usr/bin/env python3
# 4h_12h_camarilla_volume_reversal_v1
# Hypothesis: 4-hour mean reversion using 12-hour Camarilla pivot levels with volume confirmation.
# In range-bound markets (common in 2025-2026), price tends to revert from extreme Camarilla levels (H4/L4).
# Long: price crosses below L4 AND volume > 1.3x 20-period average volume.
# Short: price crosses above H4 AND volume > 1.3x 20-period average volume.
# Exit: price returns to the 12-hour daily pivot (PP) level.
# Designed for low-frequency, high-probability reversals in choppy markets with strict volume filter to limit trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour Camarilla pivot levels (based on previous 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar: based on previous bar's high, low, close
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Camarilla multipliers
    H4 = c_12h + 1.5 * (h_12h - l_12h)  # Resistance level
    L4 = c_12h - 1.5 * (h_12h - l_12h)  # Support level
    PP = (h_12h + l_12h + c_12h) / 3    # Pivot point
    
    # Align to 4h timeframe (wait for 12h bar to close)
    H4_aligned = align_htf_to_ltf(prices, df_12h, H4)
    L4_aligned = align_htf_to_ltf(prices, df_12h, L4)
    PP_aligned = align_htf_to_ltf(prices, df_12h, PP)
    
    # 20-period average volume for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(PP_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        h4 = H4_aligned[i]
        l4 = L4_aligned[i]
        pp = PP_aligned[i]
        
        vol_surge = vol > 1.3 * avg_vol
        
        if position == 1:  # Long position
            # Exit when price returns to pivot level
            if price >= pp:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price returns to pivot level
            if price <= pp:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long at L4 with volume surge
            if price < l4 and vol_surge:
                position = 1
                signals[i] = 0.25
            # Enter short at H4 with volume surge
            elif price > h4 and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals