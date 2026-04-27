#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price close above/below 1-day VWAP with volume confirmation.
# VWAP acts as dynamic support/resistance; breaks indicate institutional flow.
# Works in bull (buying pressure above VWAP) and bear (selling pressure below VWAP).
# Uses volume spike to confirm institutional participation. Target: 20-40 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP for each day
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = np.full(len(df_1d), np.nan)
    
    # Cumulative VWAP calculation
    cum_tpv = 0.0
    cum_vol = 0.0
    for i in range(len(df_1d)):
        cum_tpv += typical_price_1d[i] * volume_1d[i]
        cum_vol += volume_1d[i]
        if cum_vol > 0:
            vwap_1d[i] = cum_tpv / cum_vol
    
    # Align daily VWAP to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for VWAP and volume MA
    start_idx = max(20, 30)
    
    for i in range(start_idx, n):
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above VWAP with volume spike
            if close[i] > vwap_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below VWAP with volume spike
            elif close[i] < vwap_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses back below VWAP
            if close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above VWAP
            if close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VWAP_Breakout_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
EOF