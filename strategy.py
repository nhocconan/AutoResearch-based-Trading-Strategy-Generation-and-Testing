#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_v1
# Strategy: Camarilla pivot breakout with volume confirmation on 4h timeframe
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels from 1d act as strong support/resistance. 
# Breakouts with volume confirmation capture institutional moves. 
# Works in bull by catching breakouts above resistance, 
# and in bear by catching breakdowns below support.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Formula: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # We'll use H3 and L3 as entry levels (more sensitive)
    # H3 = C + 1.125*(H-L), L3 = C - 1.125*(H-L)
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Calculate Camarilla H3 and L3 levels
    camarilla_h3 = prev_close + 1.125 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.125 * (prev_high - prev_low)
    
    # Align to 4h timeframe (these levels are valid for the entire day)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation (20-period average)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume lookback
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current values
        price_now = close[i]
        vol_now = volume[i]
        
        # Camarilla levels for today
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        # Breakout conditions with volume confirmation
        breakout_long = (price_now > h3_level) and vol_spike[i]
        breakdown_short = (price_now < l3_level) and vol_spike[i]
        
        # Exit conditions: price returns to opposite level or mid-point
        mid_point = (h3_level + l3_level) / 2
        exit_long = position == 1 and (price_now < mid_point)
        exit_short = position == -1 and (price_now > mid_point)
        
        # Trading logic
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakdown_short and position != -1:
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