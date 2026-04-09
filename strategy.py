#!/usr/bin/env python3
# 6h_ewo_macd_volume_v1
# Hypothesis: Combines Elder's Wave Oscillator (EWO) for momentum, MACD for trend confirmation, and volume spike for validation.
# Works in bull markets via EWO > 0 + MACD histogram > 0 + volume spike (long entries).
# Works in bear markets via EWO < 0 + MACD histogram < 0 + volume spike (short entries).
# Uses 6h primary timeframe with 1d EWO and MACD to reduce noise and avoid overtrading.
# Target: 50-150 total trades over 4 years (12-37/year) with strict multi-factor confluence.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ewo_macd_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1. EWO (Elder's Wave Oscillator) from 1d: 5-period SMA - 34-period SMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma5_1d = np.full(len(close_1d), np.nan)
    sma34_1d = np.full(len(close_1d), np.nan)
    
    # Calculate SMA with proper handling
    for i in range(len(close_1d)):
        if i >= 4:  # 5-period SMA
            sma5_1d[i] = np.mean(close_1d[i-4:i+1])
        if i >= 33:  # 34-period SMA
            sma34_1d[i] = np.mean(close_1d[i-33:i+1])
    
    ewo_1d = sma5_1d - sma34_1d
    ewo_1d_aligned = align_htf_to_ltf(prices, df_1d, ewo_1d)
    
    # 2. MACD from 1d: (12 EMA - 26 EMA), Signal line = 9 EMA of MACD
    ema12_1d = np.zeros(len(close_1d))
    ema26_1d = np.zeros(len(close_1d))
    ema12_1d[0] = close_1d[0]
    ema26_1d[0] = close_1d[0]
    alpha12 = 2 / (12 + 1)
    alpha26 = 2 / (26 + 1)
    
    for i in range(1, len(close_1d)):
        ema12_1d[i] = alpha12 * close_1d[i] + (1 - alpha12) * ema12_1d[i-1]
        ema26_1d[i] = alpha26 * close_1d[i] + (1 - alpha26) * ema26_1d[i-1]
    
    macd_line_1d = ema12_1d - ema26_1d
    signal_line_1d = np.zeros(len(macd_line_1d))
    alpha9 = 2 / (9 + 1)
    signal_line_1d[0] = macd_line_1d[0]
    for i in range(1, len(macd_line_1d)):
        signal_line_1d[i] = alpha9 * macd_line_1d[i] + (1 - alpha9) * signal_line_1d[i-1]
    
    macd_hist_1d = macd_line_1d - signal_line_1d
    macd_hist_1d_aligned = align_htf_to_ltf(prices, df_1d, macd_hist_1d)
    
    # 3. Volume confirmation (20-period average)
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ewo_1d_aligned[i]) or np.isnan(macd_hist_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: EWO turns negative OR MACD histogram turns negative
            if ewo_1d_aligned[i] <= 0 or macd_hist_1d_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EWO turns positive OR MACD histogram turns positive
            if ewo_1d_aligned[i] >= 0 or macd_hist_1d_aligned[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: EWO > 0 AND MACD histogram > 0 with volume confirmation
            if (ewo_1d_aligned[i] > 0 and 
                macd_hist_1d_aligned[i] > 0 and 
                vol_ok):
                position = 1
                signals[i] = 0.25
            # Enter short: EWO < 0 AND MACD histogram < 0 with volume confirmation
            elif (ewo_1d_aligned[i] < 0 and 
                  macd_hist_1d_aligned[i] < 0 and 
                  vol_ok):
                position = -1
                signals[i] = -0.25
    
    return signals