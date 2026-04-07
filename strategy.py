#!/usr/bin/env python3
"""
6h Parabolic SAR with Weekly Trend Filter
Long when Parabolic SAR flips below price AND weekly trend is up (close > weekly open)
Short when Parabolic SAR flips above price AND weekly trend is down (close < weekly open)
Exit when SAR flips opposite direction
Weekly trend filter reduces whipsaws in ranging markets while keeping trend-following edge
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_parabolic_sar_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly trend filter (1w) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly trend: 1 if weekly close > weekly open (bullish week), -1 otherwise
    weekly_trend_raw = np.where(df_1w['close'] > df_1w['open'], 1, -1)
    weekly_trend = align_htf_to_ltf(prices, df_1w, weekly_trend_raw)
    
    # === Parabolic SAR ===
    # Initialize
    psar = np.zeros(n)
    bull = True  # True for long, False for short
    af = 0.02    # acceleration factor
    max_af = 0.2
    ep = high[0] if bull else low[0]  # extreme point
    psar[0] = low[0] if bull else high[0]
    
    # Calculate SAR
    for i in range(1, n):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR is within prior period's range
            psar[i] = min(psar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR is within prior period's range
            psar[i] = max(psar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
        
        # Reverse if price crosses SAR
        reverse = False
        if bull and low[i] < psar[i]:
            bull = False
            reverse = True
            ep = low[i]
            af = 0.02
        elif not bull and high[i] > psar[i]:
            bull = True
            reverse = True
            ep = high[i]
            af = 0.02
        
        if reverse:
            psar[i] = ep  # SAR at reversal point is the extreme point
        else:
            # Update extreme point and acceleration factor
            if bull:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(psar[i]) or np.isnan(weekly_trend[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: SAR flips above price (trend reversal)
            if psar[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: SAR flips below price (trend reversal)
            if psar[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry: SAR flip with weekly trend confirmation
            if close[i] > psar[i] and weekly_trend[i] == 1:
                # Price above SAR and weekly trend up -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < psar[i] and weekly_trend[i] == -1:
                # Price below SAR and weekly trend down -> short
                position = -1
                signals[i] = -0.25
    
    return signals