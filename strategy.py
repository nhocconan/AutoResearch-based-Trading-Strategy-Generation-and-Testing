#!/usr/bin/env python3

# 6H_PARABOLIC_SAR_REVERSAL_1D_TREND_FILTER
# Hypothesis: Parabolic SAR on 6h combined with 1d EMA200 trend filter captures reversal points in trending markets.
# In bull markets: Long when SAR flips below price and 1d trend is up.
# In bear markets: Short when SAR flips above price and 1d trend is down.
# Uses acceleration factor 0.02 and max 0.2 for responsiveness.
# Target: 20-40 trades/year on 6h timeframe to avoid overtrading.

name = "6H_PARABOLIC_SAR_REVERSAL_1D_TREND_FILTER"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA200 for 1d trend filter
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # Parabolic SAR calculation
    # Initialize arrays
    sar = np.zeros(n)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
    af = np.zeros(n)     # acceleration factor
    ep = np.zeros(n)     # extreme point
    
    # Initialize first values
    sar[0] = low[0]
    trend[0] = 1
    af[0] = 0.02
    ep[0] = high[0]
    
    # Calculate SAR for each bar
    for i in range(1, n):
        # Previous values
        prev_sar = sar[i-1]
        prev_trend = trend[i-1]
        prev_af = af[i-1]
        prev_ep = ep[i-1]
        
        # Calculate current SAR
        sar[i] = prev_sar + prev_af * (prev_ep - prev_sar)
        
        # Check for trend reversal
        if prev_trend == 1:  # Was in uptrend
            if low[i] <= sar[i]:  # Price touched or went below SAR -> reverse to downtrend
                trend[i] = -1
                sar[i] = prev_ep  # SAR becomes previous EP
                af[i] = 0.02      # Reset AF
                ep[i] = low[i]    # EP becomes current low
            else:  # Continue uptrend
                trend[i] = 1
                if high[i] > prev_ep:  # New high
                    ep[i] = high[i]
                    af[i] = min(prev_af + 0.02, 0.2)  # Increase AF up to max
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
        else:  # Was in downtrend
            if high[i] >= sar[i]:  # Price touched or went above SAR -> reverse to uptrend
                trend[i] = 1
                sar[i] = prev_ep  # SAR becomes previous EP
                af[i] = 0.02      # Reset AF
                ep[i] = high[i]   # EP becomes current high
            else:  # Continue downtrend
                trend[i] = -1
                if low[i] < prev_ep:  # New low
                    ep[i] = low[i]
                    af[i] = min(prev_af + 0.02, 0.2)  # Increase AF up to max
                else:
                    ep[i] = prev_ep
                    af[i] = prev_af
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if EMA200 not ready
        if np.isnan(ema200_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: SAR flips below price (uptrend) and 1d trend up
            if trend[i] == 1 and close[i] > sar[i] and close[i] > ema200_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: SAR flips above price (downtrend) and 1d trend down
            elif trend[i] == -1 and close[i] < sar[i] and close[i] < ema200_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: SAR flips above price (trend change to down)
            if trend[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: SAR flips below price (trend change to up)
            if trend[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals