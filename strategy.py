# SPDX-FileCopyrightText: 2025 Alpaca Trading Solutions
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot levels from weekly high-low range + volume spike confirmation
# Long when price touches L3 or L4 level with volume > 2x 20-period average
# Short when price touches H3 or H4 level with volume > 2x 20-period average
# Exit when price crosses back to H4/L3 levels (neutral zone)
# Uses weekly Camarilla for institutional levels, volume for confirmation, tight entries to minimize fees
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drag and robust performance

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Camarilla pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly high-low range for Camarilla levels
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_range = weekly_high - weekly_low
    
    # Calculate Camarilla levels based on weekly range
    # H4 = weekly_close + 1.1 * weekly_range/2
    # H3 = weekly_close + 1.1 * weekly_range/4
    # L3 = weekly_close - 1.1 * weekly_range/4
    # L4 = weekly_close - 1.1 * weekly_range/2
    weekly_close = df_weekly['close'].values
    camarilla_h4 = weekly_close + 1.1 * weekly_range / 2
    camarilla_h3 = weekly_close + 1.1 * weekly_range / 4
    camarilla_l3 = weekly_close - 1.1 * weekly_range / 4
    camarilla_l4 = weekly_close - 1.1 * weekly_range / 2
    
    # Align Camarilla levels to 12h timeframe (wait for weekly close)
    h4_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_l4)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need weekly data + buffer)
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 2.0  # Require 2x average volume
        
        if position == 0:
            # Long setup: price touches L3 or L4 with volume confirmation
            if ((abs(price - l3_aligned[i]) < 0.001 * l3_aligned[i] or 
                 abs(price - l4_aligned[i]) < 0.001 * l4_aligned[i]) and 
                vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price touches H3 or H4 with volume confirmation
            elif ((abs(price - h3_aligned[i]) < 0.001 * h3_aligned[i] or 
                   abs(price - h4_aligned[i]) < 0.001 * h4_aligned[i]) and 
                  vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back to H4 level (neutral zone)
            if price >= h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back to L4 level (neutral zone)
            if price <= l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Weekly_Volume"
timeframe = "12h"
leverage = 1.0