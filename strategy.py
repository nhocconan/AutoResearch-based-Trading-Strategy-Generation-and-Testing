#!/usr/bin/env python3
"""
4h_Parabolic_SAR_1dTrend_Volume
Hypothesis: 4h Parabolic SAR reversals in direction of 1d EMA50 trend with volume confirmation.
Uses Wilder's PSAR for trend-following entries and exits, filtered by higher timeframe trend and volume.
Designed to work in both bull and bear markets by following the 1d trend and using PSAR's adaptive stop/reverse mechanism.
Target: 20-40 trades/year on 4h to avoid excessive trading and fee drag.
"""

name = "4h_Parabolic_SAR_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Get price and volume arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Parabolic SAR (Wilder's)
    # Initialize
    psar = np.full(n, np.nan)
    bull = True  # start with long assumption
    af = 0.02    # acceleration factor
    max_af = 0.2
    ep = high[0] if bull else low[0]  # extreme point
    psar[0] = low[0] if bull else high[0]
    
    for i in range(1, n):
        # Previous SAR
        psar[i] = psar[i-1] + af * (ep - psar[i-1])
        
        # Ensure SAR stays within prior period's range
        if bull:
            psar[i] = min(psar[i], low[i-1], low[i-2] if i>=2 else low[i-1])
        else:
            psar[i] = max(psar[i], high[i-1], high[i-2] if i>=2 else high[i-1])
        
        # Check for reversal
        if bull:
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low[i]
                af = 0.02
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high[i]
                af = 0.02
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after enough data for EMA50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(psar[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above EMA50 (uptrend) AND PSAR flips to bullish (price > PSAR)
            if close[i] > ema_50_aligned[i] and close[i] > psar[i]:
                signals[i] = 0.25
                position = 1
            # Short: below EMA50 (downtrend) AND PSAR flips to bearish (price < PSAR)
            elif close[i] < ema_50_aligned[i] and close[i] < psar[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below PSAR (SAR flip to bearish)
            if close[i] < psar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above PSAR (SAR flip to bullish)
            if close[i] > psar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals