#!/usr/bin/env python3
# 12h_ParabolicSAR_Volume_Momentum
# Hypothesis: Parabolic SAR on 12h with volume momentum filter. Works in bull/bear: SAR adapts to trend, volume confirms momentum.
# Uses Parabolic SAR (step=0.02, max=0.2) and volume ratio (current/30-period average) for confirmation.
# 12h timeframe targets 20-50 trades/year to minimize fee drag.

name = "12h_ParabolicSAR_Volume_Momentum"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Parabolic SAR on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Parabolic SAR calculation
    psar = np.full(len(high_12h), np.nan)
    trend = np.full(len(high_12h), np.nan)  # 1 for uptrend, -1 for downtrend
    af = 0.02
    max_af = 0.2
    ep = 0
    
    # Initialize
    if high_12h[1] > high_12h[0]:
        trend[0] = 1
        psar[0] = low_12h[0]
        ep = high_12h[1]
    else:
        trend[0] = -1
        psar[0] = high_12h[0]
        ep = low_12h[1]
    
    for i in range(1, len(high_12h)):
        if trend[i-1] == 1:  # uptrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if low_12h[i] < psar[i]:  # trend reversal
                trend[i] = -1
                psar[i] = ep
                af = 0.02
                ep = low_12h[i]
            else:
                trend[i] = 1
                if high_12h[i] > ep:
                    ep = high_12h[i]
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if high_12h[i] > psar[i]:  # trend reversal
                trend[i] = 1
                psar[i] = ep
                af = 0.02
                ep = high_12h[i]
            else:
                trend[i] = -1
                if low_12h[i] < ep:
                    ep = low_12h[i]
                    af = min(af + 0.02, max_af)
    
    # Align Parabolic SAR to 12h timeframe (no additional delay needed)
    psar_aligned = align_htf_to_ltf(prices, df_12h, psar)
    
    # Volume momentum filter: current volume / 30-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 30:
        vol_ma[29] = np.mean(volume[0:30])
        for i in range(30, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 29 + volume[i]) / 30
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(psar_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above PSAR AND volume momentum
            if close[i] > psar_aligned[i] and volume_ratio[i] > 1.3:
                signals[i] = 0.25
                position = 1
            # Enter short: price below PSAR AND volume momentum
            elif close[i] < psar_aligned[i] and volume_ratio[i] > 1.3:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below PSAR
            if close[i] < psar_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above PSAR
            if close[i] > psar_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals