#!/usr/bin/env python3
"""
1d_1W_Weekly_Camarilla_R1S1_Breakout_Volume_Trend_Filter_v1
Hypothesis: Use weekly Camarilla pivot levels (R1, S1) on 1d timeframe with volume confirmation.
Long when price breaks above weekly R1 with volume spike.
Short when price breaks below weekly S1 with volume spike.
Exit when price returns to weekly pivot (PP) or volatility collapses.
Designed for 1d timeframe to capture medium-term trends with ~10-25 trades/year.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
Weekly pivots provide stronger support/resistance than daily, reducing false breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for Camarilla pivots
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels: based on previous week's OHLC
    # Using standard Camarilla formula: 
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # PP = (H+L+C)/3
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Calculate Camarilla levels for each week
    R1 = np.full_like(high_w, np.nan)
    S1 = np.full_like(low_w, np.nan)
    PP = np.full_like(high_w, np.nan)
    
    for i in range(1, len(high_w)):  # Start from 1 to use previous week's data
        H = high_w[i-1]
        L = low_w[i-1]
        C = close_w[i-1]
        range_val = H - L
        
        if range_val > 0:  # Avoid division by zero
            PP[i] = (H + L + C) / 3
            R1[i] = C + (range_val * 1.1 / 12)
            S1[i] = C - (range_val * 1.1 / 12)
        else:
            PP[i] = C
            R1[i] = C
            S1[i] = C
    
    # Align weekly levels to daily timeframe
    R1_aligned = align_htf_to_ltf(prices, df_weekly, R1)
    S1_aligned = align_htf_to_ltf(prices, df_weekly, S1)
    PP_aligned = align_htf_to_ltf(prices, df_weekly, PP)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(PP_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.8 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.8 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above weekly R1 with volume
            if price > R1_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below weekly S1 with volume
            elif price < S1_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly pivot or volatility collapse
            if price <= PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly pivot or volatility collapse
            if price >= PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Weekly_Camarilla_R1S1_Breakout_Volume_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0