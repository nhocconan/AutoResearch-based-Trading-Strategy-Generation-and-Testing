#!/usr/bin/env python3
# 1d_1W_Camarilla_R1_S1_Breakout
# Strategy: Use weekly Camarilla pivot levels for long-term trend direction, enter on 1d breakouts of R1/S1 with volume confirmation.
# Long when price breaks above weekly R1 with volume > 1.5x 20-period average.
# Short when price breaks below weekly S1 with volume > 1.5x 20-period average.
# Exit when price returns to weekly pivot (PP) or opposite 1/2 level.
# Weekly trend filter ensures we only trade in direction of higher timeframe momentum.
# Designed for 1d timeframe with low frequency to minimize fee drag and work in both bull and bear markets.

name = "1d_1W_Camarilla_R1_S1_Breakout"
timeframe = "1d"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels (using 1w as HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels: based on previous week's OHLC
    # Camarilla equations:
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.1 * (High - Low)
    # H2 = Close + 0.6 * (High - Low)
    # H1 = Close + 0.318 * (High - Low)
    # L1 = Close - 0.318 * (High - Low)
    # L2 = Close - 0.6 * (High - Low)
    # L3 = Close - 1.1 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # Pivot (PP) = (High + Low + Close) / 3
    
    # We use the previous week's data to avoid lookahead
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    rng = weekly_high - weekly_low
    H1 = weekly_close + 0.318 * rng
    H2 = weekly_close + 0.6 * rng
    H3 = weekly_close + 1.1 * rng
    H4 = weekly_close + 1.5 * rng
    L1 = weekly_close - 0.318 * rng
    L2 = weekly_close - 0.6 * rng
    L3 = weekly_close - 1.1 * rng
    L4 = weekly_close - 1.5 * rng
    PP = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly levels to daily timeframe (shifted by one week to avoid lookahead)
    H1_aligned = align_htf_to_ltf(prices, df_1w, H1)
    H2_aligned = align_htf_to_ltf(prices, df_1w, H2)
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1w, H4)
    L1_aligned = align_htf_to_ltf(prices, df_1w, L1)
    L2_aligned = align_htf_to_ltf(prices, df_1w, L2)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1w, L4)
    PP_aligned = align_htf_to_ltf(prices, df_1w, PP)
    
    # Volume confirmation: 20-period average volume
    vol_ma = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after enough data for weekly alignment and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if weekly data not ready
        if (np.isnan(H1_aligned[i]) or np.isnan(L1_aligned[i]) or np.isnan(PP_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signal: break above H1 with volume confirmation
            if high[i] > H1_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short signal: break below L1 with volume confirmation
            elif low[i] < L1_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to pivot or below L1 (mean reversion)
            if close[i] <= PP_aligned[i] or close[i] < L1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to pivot or above H1
            if close[i] >= PP_aligned[i] or close[i] > H1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals