#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels for today based on yesterday's range
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # H2 = close + 1.1 * (high - low) / 6
    # L2 = close - 1.1 * (high - low) / 6
    # H1 = close + 1.1 * (high - low) / 12
    # L1 = close - 1.1 * (high - low) / 12
    
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 * range_1d / 2
    camarilla_l4 = close_1d - 1.1 * range_1d / 2
    camarilla_h3 = close_1d + 1.1 * range_1d / 4
    camarilla_l3 = close_1d - 1.1 * range_1d / 4
    camarilla_h2 = close_1d + 1.1 * range_1d / 6
    camarilla_l2 = close_1d - 1.1 * range_1d / 6
    camarilla_h1 = close_1d + 1.1 * range_1d / 12
    camarilla_l1 = close_1d - 1.1 * range_1d / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_4h = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_4h = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_4h = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_4h = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Volume filter: 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if not ready
        if (np.isnan(camarilla_h4_4h[i]) or np.isnan(camarilla_l4_4h[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout signals with volume confirmation
        # Long: price breaks above H3 or H4
        long_breakout = (close[i] > camarilla_h3_4h[i] or close[i] > camarilla_h4_4h[i]) and volume_ok[i]
        # Short: price breaks below L3 or L4
        short_breakout = (close[i] < camarilla_l3_4h[i] or close[i] < camarilla_l4_4h[i]) and volume_ok[i]
        
        # Exit when price returns to opposite side
        exit_long = close[i] < camarilla_l1_4h[i]  # Return to L1 area
        exit_short = close[i] > camarilla_h1_4h[i]  # Return to H1 area
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals