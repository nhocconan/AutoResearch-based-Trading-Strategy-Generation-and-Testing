#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels from previous week
    # H3 = C + 1.1*(H-L)/4
    # L3 = C - 1.1*(H-L)/4
    # H4 = C + 1.1*(H-L)/2
    # L4 = C - 1.1*(H-L)/2
    camarilla_h3 = close_1w + 1.1 * (high_1w - low_1w) / 4
    camarilla_l3 = close_1w - 1.1 * (high_1w - low_1w) / 4
    camarilla_h4 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_l4 = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align to daily timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Volume filter - 20-period average on daily data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Camarilla breakout signals with volume confirmation
        # Long: Price breaks above H4
        long_signal = close[i] > camarilla_h4_aligned[i] and volume_ok[i]
        # Short: Price breaks below L4
        short_signal = close[i] < camarilla_l4_aligned[i] and volume_ok[i]
        
        # Exit when price returns to H3/L3 levels
        exit_long = close[i] < camarilla_h3_aligned[i]
        exit_short = close[i] > camarilla_l3_aligned[i]
        
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