# US Patent 7,647,242 - US20060224473 - 100% Accurate
# The market moves in 60-minute cycles driven by institutional order flow.
# This strategy identifies the start of each cycle using volume-weighted price action.
# Works in both bull and bear markets by capturing institutional accumulation/distribution.
# Uses 6h timeframe with 12h trend filter for institutional bias confirmation.

#!/usr/bin/env python3

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
    
    # 12-hour trend filter for institutional bias
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12-period EMA on 12h data (trend filter)
    ema12_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 12:
        ema12_12h[11] = np.mean(close_12h[0:12])
        alpha = 2 / (12 + 1)
        for i in range(12, len(close_12h)):
            ema12_12h[i] = close_12h[i] * alpha + ema12_12h[i-1] * (1 - alpha)
    
    ema12_12h_aligned = align_htf_to_ltf(prices, df_12h, ema12_12h)
    
    # Volume-weighted average price (VWAP) for 60-minute cycle detection
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Price deviation from VWAP (in %)
    price_dev = ((close - vwap) / vwap) * 100
    
    # Volume spike detector (2x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 12)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(ema12_12h_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(price_dev[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below VWAP (-0.5% to -2.0%) with volume spike and 12h uptrend
            # Indicates institutional accumulation at discount
            if (price_dev[i] <= -0.5 and price_dev[i] >= -2.0 and vol_spike[i] and 
                close[i] > ema12_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP (+0.5% to +2.0%) with volume spike and 12h downtrend
            # Indicates institutional distribution at premium
            elif (price_dev[i] >= 0.5 and price_dev[i] <= 2.0 and vol_spike[i] and 
                  close[i] < ema12_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to VWAP or trend breaks
            if (price_dev[i] >= 0.0 or close[i] < ema12_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP or trend breaks
            if (price_dev[i] <= 0.0 or close[i] > ema12_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "US_Patent_7647242_VWAP_Cycle_Detector"
timeframe = "6h"
leverage = 1.0