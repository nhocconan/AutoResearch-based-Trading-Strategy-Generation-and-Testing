#!/usr/bin/env python3
# 12h_1d_VWAP_Deviation_Reversion
# Hypothesis: On 12h timeframe, trade reversions from daily VWAP with volume confirmation.
# Uses daily VWAP deviation as mean-reversion signal. Works in both bull and bear markets
# as it fades extremes rather than following trends. Targets 15-25 trades per year.

name = "12h_1d_VWAP_Deviation_Reversion"
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
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily VWAP (Volume Weighted Average Price)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Align daily VWAP to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate daily ATR for dynamic thresholds
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First TR
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume average for confirmation (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Calculate deviation from VWAP in ATR units
        if vwap_aligned[i] != 0:
            deviation = (close[i] - vwap_aligned[i]) / atr_aligned[i]
        else:
            deviation = 0
        
        if position == 0:
            # Long when price is significantly below VWAP (oversold)
            if (deviation < -1.5 and 
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short when price is significantly above VWAP (overbought)
            elif (deviation > 1.5 and 
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to VWAP or further extension
            if deviation > -0.5:  # Return halfway to VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to VWAP or further extension
            if deviation < 0.5:  # Return halfway to VWAP
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals