#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v34"
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
    
    # Load 1d data ONCE
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's close for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = close_1d[0]  # First day uses its own close
    
    # Calculate Camarilla levels (based on previous day)
    high_low = high_1d - low_1d
    camarilla_h5 = prev_close + 1.1 * high_low / 2
    camarilla_h4 = prev_close + 1.1 * high_low / 4
    camarilla_h3 = prev_close + 1.1 * high_low / 6
    camarilla_l3 = prev_close - 1.1 * high_low / 6
    camarilla_l4 = prev_close - 1.1 * high_low / 4
    camarilla_l5 = prev_close - 1.1 * high_low / 2
    
    # Align Camarilla levels to 4h timeframe
    h5 = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4 = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3 = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3 = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4 = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5 = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Volume spike detection (1d)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    vol_current = align_htf_to_ltf(prices, df_1d, volume_1d)
    vol_spike = vol_current > 2.0 * vol_avg_aligned  # Volume > 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after 20-bar warmup
        if np.isnan(h4[i]) or np.isnan(l4[i]) or np.isnan(vol_spike[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price = close[i]
        
        # Long when price breaks above H4 with volume spike
        long_signal = price > h4[i] and vol_spike[i]
        
        # Short when price breaks below L4 with volume spike
        short_signal = price < l4[i] and vol_spike[i]
        
        # Exit when price returns to midpoint (H3/L3)
        midpoint = (h3[i] + l3[i]) / 2
        exit_long = price < midpoint
        exit_short = price > midpoint
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals