#!/usr/bin/env python3
# 6h_1d_pivots_breakout_volume_v2
# Hypothesis: Trade breakouts of daily Camarilla pivot levels with volume confirmation on 6h timeframe.
# Uses Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) from daily timeframe.
# Long when price breaks above R4 with volume surge and price above daily VWAP.
# Short when price breaks below S4 with volume surge and price below daily VWAP.
# Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Daily trend filter (price vs VWAP) ensures alignment with higher timeframe momentum, working in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_pivots_breakout_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and VWAP
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP for daily
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = np.where(vwap_den > 0, vwap_num / vwap_den, typical_price_1d)
    
    # Calculate Camarilla pivot levels for daily
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # R4 = Close + Range * 1.1/2, S4 = Close - Range * 1.1/2
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    # R3 = Close + Range * 1.1/4, S3 = Close - Range * 1.1/4
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    
    # Align daily levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below VWAP
            if close[i] < vwap_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above VWAP
            if close[i] > vwap_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above R4 with volume surge and price above daily VWAP
            if (close[i] > r4_aligned[i] and vol_surge and 
                close[i] > vwap_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S4 with volume surge and price below daily VWAP
            elif (close[i] < s4_aligned[i] and vol_surge and 
                  close[i] < vwap_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals