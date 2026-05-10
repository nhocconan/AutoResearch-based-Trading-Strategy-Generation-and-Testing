#!/usr/bin/env python3
"""
4h_Parabolic_SAR_Trend_Follow_1d_Volume
Hypothesis: Parabolic SAR trend following with 1d EMA50 trend filter and volume confirmation.
Works in both bull and bear markets by following 1d trend and using volatility-based entries.
Target: 20-30 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "4h_Parabolic_SAR_Trend_Follow_1d_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate Parabolic SAR
    sar = np.full(n, np.nan)
    af = np.full(n, np.nan)  # acceleration factor
    ep = np.full(n, np.nan)  # extreme point
    long = np.full(n, True)  # True for long, False for short
    
    # Initialize
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = 0.02
    long[0] = True
    
    for i in range(1, n):
        if long[i-1]:  # was long
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            # Reverse if price < SAR
            if low[i] < sar[i]:
                long[i] = False
                sar[i] = ep[i-1]  # SAR becomes prior EP
                ep[i] = low[i]    # EP becomes lowest low
                af[i] = 0.02      # reset AF
            else:
                long[i] = True
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + 0.02, 0.2)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # was short
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            # Reverse if price > SAR
            if high[i] > sar[i]:
                long[i] = True
                sar[i] = ep[i-1]  # SAR becomes prior EP
                ep[i] = high[i]   # EP becomes highest high
                af[i] = 0.02      # reset AF
            else:
                long[i] = False
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + 0.02, 0.2)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume SMA(20)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(sar[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Price above SAR with uptrend and volume confirmation
            if close[i] > sar[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price below SAR with downtrend and volume confirmation
            elif close[i] < sar[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below SAR
            if close[i] < sar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above SAR
            if close[i] > sar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals