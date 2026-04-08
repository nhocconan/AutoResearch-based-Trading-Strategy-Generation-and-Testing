#!/usr/bin/env python3
# 4h_camilla_pivot_breakout_volume_v1
# Hypothesis: Camarilla pivot breakouts with volume confirmation on 4h timeframe.
# Long when price breaks above R4 with volume > 1.5x average.
# Short when price breaks below S4 with volume > 1.5x average.
# Exit when price crosses the opposite pivot level (S3/R3) or volume drops below average.
# Uses Camarilla levels from 1d timeframe for structure, volume for confirmation.
# Target: 20-50 trades/year with strict entry conditions to avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camilla_pivot_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using typical formula: R4 = C + ((H-L)*1.1/2), S4 = C - ((H-L)*1.1/2)
    # where C, H, L are from previous day
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Calculate pivot levels
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align to 4h timeframe (wait for 1d bar to close)
    r4_4h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: 1.5x 20-period average (minimum period for calculation)
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = vol_ma_period + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or 
            np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below S3 or volume drops below average
            if close[i] < s3_4h[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above R3 or volume drops below average
            if close[i] > r3_4h[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above R4 with volume surge
            if (close[i] > r4_4h[i] and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below S4 with volume surge
            elif (close[i] < s4_4h[i] and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals