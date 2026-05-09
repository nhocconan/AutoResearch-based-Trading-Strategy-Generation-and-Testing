#!/usr/bin/env python3
# 1h_Camarilla_1dTrend_Volume_Confirm
# Hypothesis: Use 1d Camarilla pivot levels for trend direction (price > R1 = long bias, < S1 = short bias) and enter on 1h breakouts with volume confirmation.
# 1h is used only for entry timing to reduce false breakouts. Targets 15-35 trades/year per symbol.
# Works in bull/bear by following higher timeframe trend. Volume filter reduces false signals.

name = "1h_Camarilla_1dTrend_Volume_Confirm"
timeframe = "1h"
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
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    daily_range = high_1d - low_1d
    camarilla_R1 = close_1d + daily_range * 1.1 / 12
    camarilla_S1 = close_1d - daily_range * 1.1 / 12
    
    # Align daily Camarilla levels to 1h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume filter: volume > 1.5 * 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price > R1 = long bias, price < S1 = short bias
        long_bias = close[i] > camarilla_R1_aligned[i]
        short_bias = close[i] < camarilla_S1_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above R1 AND volume confirmation AND long bias
            if long_bias and volume_ratio[i] > 1.5:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S1 AND volume confirmation AND short bias
            elif short_bias and volume_ratio[i] > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S1 (trend reversal)
            if short_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price breaks above R1 (trend reversal)
            if long_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals