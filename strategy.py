#!/usr/bin/env python3
"""
1d_1w_Parabolic_SAR_Trend_Follow
Hypothesis: Uses weekly Parabolic SAR for trend direction and daily price action for entries, with volume confirmation. Designed to work in both bull and bear markets by following the weekly trend and avoiding counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Parabolic SAR
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Parabolic SAR
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Initialize SAR
    psar = np.zeros_like(close_1w)
    trend = np.zeros_like(close_1w)  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    
    # Set initial values
    if close_1w[1] > close_1w[0]:
        trend[0] = 1
        psar[0] = low_1w[0]
        ep = high_1w[0]  # extreme point
    else:
        trend[0] = -1
        psar[0] = high_1w[0]
        ep = low_1w[0]
    
    # Calculate SAR for each week
    for i in range(1, len(close_1w)):
        if trend[i-1] == 1:  # uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR is below low
            if psar[i] > low_1w[i]:
                trend[i] = -1
                psar[i] = ep
                ep = low_1w[i]
                af = 0.02
            else:
                trend[i] = 1
                if high_1w[i] > ep:
                    ep = high_1w[i]
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR is above high
            if psar[i] < high_1w[i]:
                trend[i] = 1
                psar[i] = ep
                ep = high_1w[i]
                af = 0.02
            else:
                trend[i] = -1
                if low_1w[i] < ep:
                    ep = low_1w[i]
                    af = min(af + 0.02, max_af)
    
    # Align weekly SAR and trend to daily timeframe
    psar_aligned = align_htf_to_ltf(prices, df_1w, psar)
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(psar_aligned[i]) or np.isnan(trend_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long when weekly trend is up and price above SAR with volume
            if trend_aligned[i] == 1 and close[i] > psar_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short when weekly trend is down and price below SAR with volume
            elif trend_aligned[i] == -1 and close[i] < psar_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below SAR or trend changes
            if close[i] < psar_aligned[i] or trend_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above SAR or trend changes
            if close[i] > psar_aligned[i] or trend_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Parabolic_SAR_Trend_Follow"
timeframe = "1d"
leverage = 1.0