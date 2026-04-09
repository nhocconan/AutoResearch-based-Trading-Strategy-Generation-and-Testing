#!/usr/bin/env python3
# 4h_vwap_std_breakout_v1
# Hypothesis: Price breaking above/below VWAP + 1 standard deviation on 4h timeframe with volume confirmation.
# VWAP bands act as dynamic support/resistance. Breakouts with high volume indicate strong momentum.
# Works in bull/bear markets by capturing momentum shifts at key levels.
# Target: 20-40 trades/year (80-160 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_vwap_std_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Typical Price
    tp = (high + low + close) / 3.0
    
    # VWAP calculation (cumulative)
    vwap = np.zeros(n)
    tpv_sum = 0.0
    vol_sum = 0.0
    for i in range(n):
        tpv_sum += tp[i] * volume[i]
        vol_sum += volume[i]
        if vol_sum > 0:
            vwap[i] = tpv_sum / vol_sum
        else:
            vwap[i] = tp[i]
    
    # Calculate standard deviation of price from VWAP
    squared_diff = (tp - vwap) ** 2
    variance = np.zeros(n)
    var_sum = 0.0
    for i in range(n):
        var_sum += squared_diff[i] * volume[i]
        if vol_sum > 0:
            variance[i] = var_sum / vol_sum
        else:
            variance[i] = 0.0
    std_dev = np.sqrt(variance)
    
    # Upper and lower bands (VWAP ± 1 std dev)
    upper_band = vwap + std_dev
    lower_band = vwap - std_dev
    
    # Volume confirmation (20-period average)
    vol_ma_20 = np.zeros(n)
    vol_sum_ma = 0
    for i in range(n):
        vol_sum_ma += volume[i]
        if i >= 20:
            vol_sum_ma -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum_ma / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(vwap[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price closes below VWAP
            if close[i] < vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above VWAP
            if close[i] > vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above upper band with volume confirmation
            if close[i] > upper_band[i] and vol_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below lower band with volume confirmation
            elif close[i] < lower_band[i] and vol_ok:
                position = -1
                signals[i] = -0.25
    
    return signals