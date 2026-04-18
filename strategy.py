#!/usr/bin/env python3
"""
12h_ParabolicSAR_Volume_Trend
Hypothesis: Parabolic SAR combined with volume spike and 1-day EMA50 trend filter captures 
breakouts from strong trends while filtering out false signals. Works in both bull and bear 
markets by following the trend direction. Target: 20-40 trades/year on 12h timeframe.
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
    volume = prices['volume'].values
    
    # Calculate Parabolic SAR
    # Parameters: af_start=0.02, af_increment=0.02, af_max=0.2
    psar = np.zeros(n)
    psar_up = np.zeros(n)  # SAR in uptrend (below price)
    psar_down = np.zeros(n)  # SAR in downtrend (above price)
    
    # Initialize
    psar[0] = low[0]
    psar_up[0] = low[0]
    psar_down[0] = high[0]
    
    # Trend: 1 for uptrend, -1 for downtrend
    trend = np.ones(n)
    # Acceleration factor
    af = np.zeros(n)
    af[0] = 0.02
    # Extreme point
    ep = np.zeros(n)
    ep[0] = high[0]
    
    for i in range(1, n):
        if trend[i-1] == 1:  # was uptrend
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            # Check for reversal
            if low[i] < psar[i]:
                trend[i] = -1
                psar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = low[i]
                af[i] = 0.02
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + 0.02, 0.2)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # was downtrend
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            # Check for reversal
            if high[i] > psar[i]:
                trend[i] = 1
                psar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = high[i]
                af[i] = 0.02
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + 0.02, 0.2)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        
        # Store SAR values for plotting (not used in logic)
        psar_up[i] = psar[i] if trend[i] == 1 else np.nan
        psar_down[i] = psar[i] if trend[i] == -1 else np.nan
    
    # 1-day EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    k = 2 / (50 + 1)
    for i in range(50, len(close_1d)):
        if i == 50:
            ema50_1d[i] = np.mean(close_1d[0:51])
        else:
            ema50_1d[i] = close_1d[i] * k + ema50_1d[i-1] * (1 - k)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above PSAR (uptrend signal) with volume spike and 1-day uptrend
            if (close[i] > psar[i] and vol_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below PSAR (downtrend signal) with volume spike and 1-day downtrend
            elif (close[i] < psar[i] and vol_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below PSAR or 1-day trend turns down
            if (close[i] < psar[i] or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above PSAR or 1-day trend turns up
            if (close[i] > psar[i] or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ParabolicSAR_Volume_Trend"
timeframe = "12h"
leverage = 1.0