#!/usr/bin/env python3
# 12h_ParabolicSAR_1dTrend_Filter
# Hypothesis: Parabolic SAR on 12h captures trend reversals with tight stop-loss, filtered by 1d EMA50 trend direction.
# Only take longs when price > 1d EMA50 (uptrend) and SAR flips below price; shorts when price < 1d EMA50 (downtrend) and SAR flips above price.
# Uses Parabolic SAR's built-in stop and reversal mechanism to minimize whipsaw and trade frequency.
# Designed for low trade frequency (12-37/year) with strong trend capture in both bull and bear markets.

name = "12h_ParabolicSAR_1dTrend_Filter"
timeframe = "12h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Parabolic SAR parameters
    af_start = 0.02
    af_increment = 0.02
    af_max = 0.2
    
    # Initialize arrays
    sar = np.full(n, np.nan)
    trend = np.full(n, np.nan)  # 1 for uptrend, -1 for downtrend
    af = np.full(n, np.nan)
    ep = np.full(n, np.nan)  # extreme point
    
    # Initialize first values
    sar[0] = low[0]
    trend[0] = 1  # start assuming uptrend
    af[0] = af_start
    ep[0] = high[0]
    
    # Calculate SAR for each bar
    for i in range(1, n):
        if trend[i-1] == 1:  # was in uptrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            # Check for trend reversal
            if low[i] <= sar[i]:
                trend[i] = -1  # reverse to downtrend
                sar[i] = ep[i-1]  # SAR becomes prior EP
                ep[i] = low[i]    # reset EP to current low
                af[i] = af_start  # reset acceleration factor
            else:
                trend[i] = 1  # remain in uptrend
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # was in downtrend
            sar[i] = sar[i-1] + af[i-1] * (sar[i-1] - ep[i-1])
            # Check for trend reversal
            if high[i] >= sar[i]:
                trend[i] = 1  # reverse to uptrend
                sar[i] = ep[i-1]  # SAR becomes prior EP
                ep[i] = high[i]   # reset EP to current high
                af[i] = af_start  # reset acceleration factor
            else:
                trend[i] = -1  # remain in downtrend
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # Align daily EMA50 trend to 12h timeframe (already done above)
    # Now generate signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # SAR needs at least 2 points
    
    for i in range(start_idx, n):
        # Skip if EMA trend not ready
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: SAR indicates uptrend AND price > 1d EMA50 (uptrend filter)
            if trend[i] == 1 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: SAR indicates downtrend AND price < 1d EMA50 (downtrend filter)
            elif trend[i] == -1 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: SAR flips to downtrend
            if trend[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: SAR flips to uptrend
            if trend[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals