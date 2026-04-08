#!/usr/bin/env python3
# 12h_camilla_pivot_volume_v1
# Hypothesis: Uses 1d Camarilla pivot levels (H4/L4) with volume confirmation on 12h timeframe.
# Long when price crosses above H4 with volume > 1.5x average.
# Short when price crosses below L4 with volume > 1.5x average.
# Exit when price returns to pivot (H3/L3) or volume drops below average.
# Uses Camarilla levels for intraday support/resistance, effective in both trending and ranging markets.
# Volume surge confirms breakout strength. Target: 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    camarilla_H4 = np.full(len(close_1d), np.nan)
    camarilla_L4 = np.full(len(close_1d), np.nan)
    camarilla_H3 = np.full(len(close_1d), np.nan)
    camarilla_L3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if not np.isnan(high_1d[i-1]) and not np.isnan(low_1d[i-1]) and not np.isnan(close_1d[i-1]):
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            range_val = prev_high - prev_low
            
            camarilla_H4[i] = prev_close + range_val * 1.1 / 2
            camarilla_L4[i] = prev_close - range_val * 1.1 / 2
            camarilla_H3[i] = prev_close + range_val * 1.1 / 4
            camarilla_L3[i] = prev_close - range_val * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = vol_ma_period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma[i]) or np.isnan(camarilla_H4_aligned[i]) or 
            np.isnan(camarilla_L4_aligned[i]) or np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below H3 or volume drops below average
            if close[i] < camarilla_H3_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above L3 or volume drops below average
            if close[i] > camarilla_L3_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price crosses above H4 with volume surge
            if (close[i] > camarilla_H4_aligned[i] and 
                close[i-1] <= camarilla_H4_aligned[i-1] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price crosses below L4 with volume surge
            elif (close[i] < camarilla_L4_aligned[i] and 
                  close[i-1] >= camarilla_L4_aligned[i-1] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals