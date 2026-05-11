#!/usr/bin/env python3
name = "4h_Parabolic_SAR_Trend_Stop"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_psar(high, low, close, acceleration=0.02, max_acceleration=0.2):
    """Calculate Parabolic SAR"""
    n = len(close)
    psar = np.full(n, np.nan)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
    ep = np.zeros(n)     # extreme point
    af = np.full(n, acceleration)  # acceleration factor
    
    # Initialize
    if n < 2:
        return psar
    
    # Start with trend based on first two periods
    if close[1] > close[0]:
        trend[0] = 1
        psar[0] = low[0]
        ep[0] = high[1]
    else:
        trend[0] = -1
        psar[0] = high[0]
        ep[0] = low[1]
    
    for i in range(1, n):
        # Calculate SAR
        psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
        
        # Check if we need to reverse trend
        if trend[i-1] == 1:  # uptrend
            if low[i] <= psar[i]:  # trend reversal
                trend[i] = -1
                psar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = low[i]
                af[i] = acceleration
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + acceleration, max_acceleration)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # downtrend
            if high[i] >= psar[i]:  # trend reversal
                trend[i] = 1
                psar[i] = ep[i-1]  # SAR becomes previous EP
                ep[i] = high[i]
                af[i] = acceleration
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + acceleration, max_acceleration)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    return psar

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate Parabolic SAR on 4h data
    psar = calculate_psar(high, low, close, acceleration=0.02, max_acceleration=0.2)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(psar[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above PSAR AND above 1d EMA20 (uptrend) AND volume spike
            if close[i] > psar[i] and close[i] > ema_20_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below PSAR AND below 1d EMA20 (downtrend) AND volume spike
            elif close[i] < psar[i] and close[i] < ema_20_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below PSAR OR below 1d EMA20 (trend change)
            if close[i] < psar[i] or close[i] < ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above PSAR OR above 1d EMA20 (trend change)
            if close[i] > psar[i] or close[i] > ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals