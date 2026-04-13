#!/usr/bin/env python3
"""
6h_12h_1d_Camarilla_Reversal_Momentum
- 6h primary timeframe
- 12h Camarilla pivot levels for structure (S3/S4, R3/R4)
- 1d momentum filter (RSI(14) > 50 for long, < 50 for short)
- Entry: Price rejects S3/R3 with momentum confirmation → fade
- Exit: Price reaches S4/R4 or momentum reverses
- Position size: 0.25 (25% of capital)
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels (using previous day's range)
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    #          H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    #          H2 = C + 1.1*(H-L)/6, L2 = C - 1.1*(H-L)/6
    #          H1 = C + 1.1*(H-L)/12, L1 = C - 1.1*(H-L)/12
    # We'll use H3/L3 (R3/S3) and H4/L4 (R4/S4)
    range_12h = high_12h - low_12h
    camarilla_h4 = close_12h + 1.1 * range_12h / 2
    camarilla_l4 = close_12h - 1.1 * range_12h / 2
    camarilla_h3 = close_12h + 1.1 * range_12h / 4
    camarilla_l3 = close_12h - 1.1 * range_12h / 4
    
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # Get 1d data for momentum filter (RSI)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Momentum filter
        rsi_bullish = rsi_1d_aligned[i] > 50
        rsi_bearish = rsi_1d_aligned[i] < 50
        
        # Fade at S3/R3 with momentum confirmation
        # Long: price near S3 and bullish momentum
        long_setup = (close[i] <= camarilla_l3_aligned[i] * 1.005) and rsi_bullish  # Within 0.5% of S3
        # Short: price near R3 and bearish momentum
        short_setup = (close[i] >= camarilla_h3_aligned[i] * 0.995) and rsi_bearish  # Within 0.5% of R3
        
        # Exit conditions
        exit_long = position == 1 and (
            close[i] >= camarilla_h4_aligned[i] or  # Hit S4/R4 target
            rsi_1d_aligned[i] < 40  # Momentum reversal
        )
        exit_short = position == -1 and (
            close[i] <= camarilla_l4_aligned[i] or  # Hit S4/R4 target
            rsi_1d_aligned[i] > 60  # Momentum reversal
        )
        
        # Execute signals
        if long_setup and position != 1:
            position = 1
            signals[i] = position_size
        elif short_setup and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_Camarilla_Reversal_Momentum"
timeframe = "6h"
leverage = 1.0