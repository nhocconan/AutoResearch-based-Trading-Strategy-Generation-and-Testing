#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
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
    
    # Get 1d data for Camarilla pivot calculation (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels based on previous day's data
    # Camarilla formulas: 
    # H4 = close + 1.1*(high-low)*1.1/2
    # L4 = close - 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4
    # H2 = close + 1.1*(high-low)*1.1/6
    # L2 = close - 1.1*(high-low)*1.1/6
    # H1 = close + 1.1*(high-low)*1.1/12
    # L1 = close - 1.1*(high-low)*1.1/12
    
    # Calculate range for previous day
    prev_high = np.roll(high_1d, 1)  # Previous day's high
    prev_low = np.roll(low_1d, 1)    # Previous day's low
    prev_close = np.roll(close_1d, 1) # Previous day's close
    
    # Avoid first element (no previous day)
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else 0
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else 0
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else 0
    
    # Calculate Camarilla levels for previous day
    range_1d = prev_high - prev_low
    H4 = prev_close + 1.1 * range_1d * 1.1 / 2
    L4 = prev_close - 1.1 * range_1d * 1.1 / 2
    H3 = prev_close + 1.1 * range_1d * 1.1 / 4
    L3 = prev_close - 1.1 * range_1d * 1.1 / 4
    H2 = prev_close + 1.1 * range_1d * 1.1 / 6
    L2 = prev_close - 1.1 * range_1d * 1.1 / 6
    H1 = prev_close + 1.1 * range_1d * 1.1 / 12
    L1 = prev_close - 1.1 * range_1d * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume filter - 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout signals with volume confirmation
        # Long: break above H3 (strong resistance)
        long_signal = close[i] > H3_aligned[i] and volume_ok[i]
        # Short: break below L3 (strong support)
        short_signal = close[i] < L3_aligned[i] and volume_ok[i]
        
        # Exit when price returns to opposite level
        exit_long = close[i] < L3_aligned[i]
        exit_short = close[i] > H3_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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