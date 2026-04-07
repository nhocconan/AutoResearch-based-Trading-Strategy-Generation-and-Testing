#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: On 12-hour timeframe, use Camarilla pivot levels from daily timeframe for mean reversion entries. 
Enter long when price touches S1/S2 support with volume confirmation; enter short when price touches R1/R2 resistance with volume confirmation.
Exit when price returns to pivot point or reverses direction. Camarilla levels provide precise support/resistance in ranging markets, while volume confirmation filters false breaks.
Works in both bull and bear markets by adapting to ranging conditions that occur during consolidation phases.
Target: 20-30 trades/year to minimize fee drag while capturing mean reversion opportunities.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 12)
    # S2 = C - (Range * 1.1 / 6)
    # R1 = C + (Range * 1.1 / 12)
    # R2 = C + (Range * 1.1 / 6)
    
    d_pivot = (d_high + d_low + d_close) / 3
    d_range = d_high - d_low
    d_s1 = d_close - (d_range * 1.1 / 12)
    d_s2 = d_close - (d_range * 1.1 / 6)
    d_r1 = d_close + (d_range * 1.1 / 12)
    d_r2 = d_close + (d_range * 1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, d_pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, d_s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, d_s2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, d_r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, d_r2)
    
    # Volume filter: 12h volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for volume MA
        # Skip if any Camarilla level not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(r2_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price action
        price = close[i]
        vol_confirmed = vol_ratio[i] > 1.3
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price returns to pivot or above
            if price >= pivot_aligned[i]:
                exit_long = True
            # Exit when price breaks below S2 (stop loss)
            elif price < s2_aligned[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price returns to pivot or below
            if price <= pivot_aligned[i]:
                exit_short = True
            # Exit when price breaks above R2 (stop loss)
            elif price > r2_aligned[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price near S1 or S2 with volume confirmation
            near_s1 = abs(price - s1_aligned[i]) / s1_aligned[i] < 0.005  # Within 0.5%
            near_s2 = abs(price - s2_aligned[i]) / s2_aligned[i] < 0.005  # Within 0.5%
            long_entry = (near_s1 or near_s2) and vol_confirmed
            
            # Short entry: price near R1 or R2 with volume confirmation
            near_r1 = abs(price - r1_aligned[i]) / r1_aligned[i] < 0.005  # Within 0.5%
            near_r2 = abs(price - r2_aligned[i]) / r2_aligned[i] < 0.005  # Within 0.5%
            short_entry = (near_r1 or near_r2) and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals