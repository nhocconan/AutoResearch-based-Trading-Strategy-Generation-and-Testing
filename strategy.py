#!/usr/bin/env python3
"""
4h_Equity_Volume_Pressure_v1
Hypothesis: Combines equity curve momentum (price above/below 1d VWAP) with volume pressure 
to identify institutional accumulation/distribution. In bull markets, buying pressure 
above VWAP sustains uptrends; in bear markets, selling pressure below VWAP confirms 
continuation. Uses 1d VWAP as dynamic support/resistance and volume spike for confirmation.
Target: 20-50 trades per year on 4h timeframe.
"""

name = "4h_Equity_Volume_Pressure_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # === 1D Data for VWAP Calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate typical price and VWAP for 1d
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    cum_tpv_1d = np.cumsum(typical_price_1d * df_1d['volume'].values)
    cum_vol_1d = np.cumsum(df_1d['volume'].values)
    vwap_1d = np.divide(cum_tpv_1d, cum_vol_1d, out=np.full_like(cum_tpv_1d, np.nan), where=cum_vol_1d!=0)
    
    # Align VWAP to 4h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === Volume Spike Detection (4h) ===
    vol_ma_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma_20[:10] = np.nan
    vol_ma_20[-10:] = np.nan
    vol_ratio = np.divide(volume, vol_ma_20, out=np.full_like(volume, np.nan), where=vol_ma_20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above VWAP with volume accumulation (bullish pressure)
            if close[i] > vwap_1d_aligned[i] and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP with volume distribution (bearish pressure)
            elif close[i] < vwap_1d_aligned[i] and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below VWAP or volume dries up
            if close[i] < vwap_1d_aligned[i] or vol_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above VWAP or volume dries up
            if close[i] > vwap_1d_aligned[i] or vol_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals