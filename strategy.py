#!/usr/bin/env python3
name = "4h_Parabolic_SAR_Trend_12hATR_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 12h data once for ATR filter
    df_12h = get_htf_data(prices, '12h')
    
    # Parabolic SAR parameters
    start = 0.02
    increment = 0.02
    maximum = 0.2
    
    # Initialize arrays
    sar = np.zeros(n)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
    ep = np.zeros(n)     # extreme point
    af = np.zeros(n)     # acceleration factor
    
    # Initialize first values
    if low[0] < high[0]:
        trend[0] = 1
        sar[0] = low[0]
        ep[0] = high[0]
        af[0] = start
    else:
        trend[0] = -1
        sar[0] = high[0]
        ep[0] = low[0]
        af[0] = start
    
    # Calculate Parabolic SAR
    for i in range(1, n):
        # SAR for current period
        sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
        
        # Check for trend reversal
        if trend[i-1] == 1:  # uptrend
            if low[i] <= sar[i]:  # trend reversal to downtrend
                trend[i] = -1
                sar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = low[i]    # new EP is current low
                af[i] = start     # reset AF
            else:  # uptrend continues
                trend[i] = 1
                if high[i] > ep[i-1]:  # new high
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + increment, maximum)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # downtrend
            if high[i] >= sar[i]:  # trend reversal to uptrend
                trend[i] = 1
                sar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = high[i]   # new EP is current high
                af[i] = start     # reset AF
            else:  # downtrend continues
                trend[i] = -1
                if low[i] < ep[i-1]:  # new low
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + increment, maximum)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # 12h ATR for volatility filter (use previous completed 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Pad first element
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14)
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # ATR threshold: only trade when volatility is above average
    atr_ma = pd.Series(atr_12h_aligned).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_12h_aligned > atr_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sar[i]) or np.isnan(trend[i]) or 
            np.isnan(atr_12h_aligned[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: SAR below price (uptrend) + volatility filter
            if trend[i] == 1 and close[i] > sar[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: SAR above price (downtrend) + volatility filter
            elif trend[i] == -1 and close[i] < sar[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend turns down
            if trend[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend turns up
            if trend[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals