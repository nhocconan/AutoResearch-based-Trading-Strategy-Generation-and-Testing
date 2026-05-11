#!/usr/bin/env python3
name = "6h_ParabolicSAR_Trend_12hTrend_Filter"
timeframe = "6h"
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
    
    # Get 12h data for trend filter (12h EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Parabolic SAR parameters
    start = 0.02
    increment = 0.02
    maximum = 0.2
    
    # Initialize arrays
    sar = np.zeros(n)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
    af = np.zeros(n)     # acceleration factor
    ep = np.zeros(n)     # extreme point
    
    # Initial values
    sar[0] = low[0]
    trend[0] = 1
    af[0] = start
    ep[0] = high[0]
    
    # Calculate Parabolic SAR
    for i in range(1, n):
        # SAR formula
        sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
        
        # Determine trend and EP
        if trend[i-1] == 1:  # Uptrend
            if low[i] <= sar[i]:  # Trend reversal
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = start
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + increment, maximum)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # Downtrend
            if high[i] >= sar[i]:  # Trend reversal
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = start
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + increment, maximum)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        
        # Ensure SAR stays within bounds
        if trend[i] == 1:
            sar[i] = min(sar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
        else:
            sar[i] = max(sar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: SAR below price (uptrend signal) AND above 12h EMA50 (long-term uptrend) AND volume surge
            if sar[i] < close[i] and close[i] > ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: SAR above price (downtrend signal) AND below 12h EMA50 (long-term downtrend) AND volume surge
            elif sar[i] > close[i] and close[i] < ema_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: SAR above price (trend reversal) OR below 12h EMA50 (long-term trend change)
            if sar[i] > close[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: SAR below price (trend reversal) OR above 12h EMA50 (long-term trend change)
            if sar[i] < close[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals