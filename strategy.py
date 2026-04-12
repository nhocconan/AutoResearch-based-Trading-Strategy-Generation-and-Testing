#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_volume"
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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # H4 = C + 1.1 * (H - L) / 2  (resistance)
    # L4 = C - 1.1 * (H - L) / 2  (support)
    camarilla_high = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_low = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align to 4h timeframe (previous day's levels available after daily close)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Volume filter: 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price touches Camarilla levels with volume confirmation
        # Long when price touches or goes below L4 support with volume
        long_signal = low[i] <= camarilla_low_aligned[i] and volume_ok[i]
        # Short when price touches or goes above H4 resistance with volume
        short_signal = high[i] >= camarilla_high_aligned[i] and volume_ok[i]
        
        # Exit when price moves back inside the range
        exit_long = high[i] > camarilla_low_aligned[i]  # price back above support
        exit_short = low[i] < camarilla_high_aligned[i]  # price back below resistance
        
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