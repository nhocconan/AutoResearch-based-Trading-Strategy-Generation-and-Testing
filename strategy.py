#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_aw_oscillator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Awesome Oscillator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate median price for 1d data
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Awesome Oscillator: SMA(5) - SMA(34) of median price
    sma5 = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    sma34 = pd.Series(median_price_1d).rolling(window=34, min_periods=34).mean().values
    ao_1d = sma5 - sma34
    
    # Awesome Oscillator signal line: SMA(5) of AO
    ao_signal = pd.Series(ao_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align AO and signal line to 6h timeframe
    ao_aligned = align_htf_to_ltf(prices, df_1d, ao_1d)
    ao_signal_aligned = align_htf_to_ltf(prices, df_1d, ao_signal)
    
    # Volume filter on 6h: current volume > 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(ao_aligned[i]) or np.isnan(ao_signal_aligned[i]) or
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: AO crosses above signal line (bullish momentum) with volume
        long_signal = (ao_aligned[i] > ao_signal_aligned[i] and 
                      ao_aligned[i-1] <= ao_signal_aligned[i-1] and
                      volume_ok[i])
        
        # Short: AO crosses below signal line (bearish momentum) with volume
        short_signal = (ao_aligned[i] < ao_signal_aligned[i] and 
                       ao_aligned[i-1] >= ao_signal_aligned[i-1] and
                       volume_ok[i])
        
        # Exit when AO crosses zero (momentum change)
        exit_long = ao_aligned[i] < 0 and position == 1
        exit_short = ao_aligned[i] > 0 and position == -1
        
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