#!/usr/bin/env python3
name = "12h_1d_ParabolicSAR_Trend_Filter"
timeframe = "12h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily Parabolic SAR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Initialize SAR
    sar = np.zeros(len(high_1d))
    trend = np.ones(len(high_1d))  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    ep = high_1d[0]  # extreme point
    
    sar[0] = low_1d[0]
    trend[0] = 1
    
    for i in range(1, len(high_1d)):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if low_1d[i] < sar[i]:  # trend reversal
                trend[i] = -1
                sar[i] = ep
                ep = low_1d[i]
                af = 0.02
            else:
                trend[i] = 1
                if high_1d[i] > ep:
                    ep = high_1d[i]
                    af = min(af + 0.02, max_af)
        else:  # downtrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            if high_1d[i] > sar[i]:  # trend reversal
                trend[i] = 1
                sar[i] = ep
                ep = high_1d[i]
                af = 0.02
            else:
                trend[i] = -1
                if low_1d[i] < ep:
                    ep = low_1d[i]
                    af = min(af + 0.02, max_af)
    
    # Align daily SAR and trend to 12h timeframe
    sar_aligned = align_htf_to_ltf(prices, df_1d, sar)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend)
    
    # 12h Parabolic SAR for entry timing
    # Initialize 12h SAR
    sar_12h = np.zeros(n)
    trend_12h = np.ones(n)
    af_12h = 0.02
    max_af_12h = 0.2
    ep_12h = high[0]
    
    sar_12h[0] = low[0]
    trend_12h[0] = 1
    
    for i in range(1, n):
        if trend_12h[i-1] == 1:  # uptrend
            sar_12h[i] = sar_12h[i-1] + af_12h * (ep_12h - sar_12h[i-1])
            if low[i] < sar_12h[i]:  # trend reversal
                trend_12h[i] = -1
                sar_12h[i] = ep_12h
                ep_12h = low[i]
                af_12h = 0.02
            else:
                trend_12h[i] = 1
                if high[i] > ep_12h:
                    ep_12h = high[i]
                    af_12h = min(af_12h + 0.02, max_af_12h)
        else:  # downtrend
            sar_12h[i] = sar_12h[i-1] + af_12h * (ep_12h - sar_12h[i-1])
            if high[i] > sar_12h[i]:  # trend reversal
                trend_12h[i] = 1
                sar_12h[i] = ep_12h
                ep_12h = high[i]
                af_12h = 0.02
            else:
                trend_12h[i] = -1
                if low[i] < ep_12h:
                    ep_12h = low[i]
                    af_12h = min(af_12h + 0.02, max_af_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient data
    
    for i in range(start_idx, n):
        if (np.isnan(sar_aligned[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(sar_12h[i]) or np.isnan(trend_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: daily uptrend AND 12h price above SAR
            if trend_aligned[i] == 1 and close[i] > sar_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: daily downtrend AND 12h price below SAR
            elif trend_aligned[i] == -1 and close[i] < sar_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: daily trend turns down OR 12h price crosses below SAR
            if trend_aligned[i] == -1 or close[i] < sar_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: daily trend turns up OR 12h price crosses above SAR
            if trend_aligned[i] == 1 or close[i] > sar_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Parabolic SAR with daily trend filter
# - Daily Parabolic SAR determines the higher-timeframe trend (1d)
# - 12h Parabolic SAR provides entry/exit signals within that trend
# - Only take longs when daily trend is up (SAR below price) and 12h price above its SAR
# - Only take shorts when daily trend is down (SAR above price) and 12h price below its SAR
# - Exit when either the daily trend changes or the 12h price crosses its SAR
# - This dual-timeframe approach reduces whipsaws and works in both bull and bear markets
# - Position size 0.25 limits risk and keeps trade frequency moderate (target: 15-35 trades/year)
# - Parabolic SAR is effective in trending markets and provides automatic trailing stops