#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    6h strategy: Camarilla pivot from 1d + weekly trend + volume confirmation
    Long: price crosses above H4 with upward weekly trend and volume > avg volume
    Short: price crosses below L4 with downward weekly trend and volume > avg volume
    Exit: price touches H3/L3 or opposite pivot level
    Designed for 6h timeframe with ~12-37 trades/year
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1w linear regression slope (5-period) for long-term trend
    close_1w = df_1w['close'].values
    x = np.arange(5)
    slope_1w = np.zeros_like(close_1w, dtype=float)
    for i in range(4, len(close_1w)):
        y = close_1w[i-4:i+1]
        if np.any(np.isnan(y)):
            slope_1w[i] = np.nan
        else:
            slope = np.polyfit(x, y, 1)[0]
            slope_1w[i] = slope
    slope_1w_aligned = align_htf_to_ltf(prices, df_1w, slope_1w)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2, H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    camarilla_h4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_l4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d average volume (20-period) for volume filter
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_val = prices['volume'].iloc[i]
        h4_val = camarilla_h4_aligned[i]
        l4_val = camarilla_l4_aligned[i]
        h3_val = camarilla_h3_aligned[i]
        l3_val = camarilla_l3_aligned[i]
        slope_val = slope_1w_aligned[i]
        avg_vol_val = avg_vol_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(h4_val) or np.isnan(l4_val) or np.isnan(h3_val) or 
            np.isnan(l3_val) or np.isnan(slope_val) or np.isnan(avg_vol_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above H4 with upward weekly trend and volume > avg volume
            if (close_val > h4_val and 
                slope_val > 0 and 
                vol_val > avg_vol_val):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below L4 with downward weekly trend and volume > avg volume
            elif (close_val < l4_val and 
                  slope_val < 0 and 
                  vol_val > avg_vol_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches H3 or crosses below L4 (reversal)
            if close_val >= h3_val or close_val < l4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches L3 or crosses above H4 (reversal)
            if close_val <= l3_val or close_val > h4_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_Camarilla_Pivot_WeeklyTrend_VolumeFilter_V1
# Uses 1-day Camarilla pivot levels (H4/L4 for entry, H3/L3 for exit)
# Uses 1-week linear regression slope for trend filter
# Requires volume > 20-period average for confirmation
# Designed for 6h timeframe with ~12-37 trades/year
name = "6h_Camarilla_Pivot_WeeklyTrend_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0