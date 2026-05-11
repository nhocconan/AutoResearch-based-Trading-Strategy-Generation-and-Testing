#!/usr/bin/env python3
"""
4h_VWAP_Pullback_Volume_Spike_Trend
Hypothesis: Enters long when price pulls back to VWAP in an uptrend with volume spike.
Enters short when price pulls back to VWAP in a downtrend with volume spike.
Uses VWAP as dynamic support/resistance, volume spike for conviction, and trend filter.
Designed for low trade frequency (20-40 trades/year) to minimize fee drift.
"""

name = "4h_VWAP_Pullback_Volume_Spike_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate VWAP
    typical_price = (high + low + close) / 3.0
    tpv = typical_price * volume
    cum_tpv = np.nancumsum(tpv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_tpv, cum_vol, out=np.full_like(tpv, np.nan), where=cum_vol!=0)
    
    # 1D data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price near VWAP (within 0.5%), uptrend, volume spike
            price_vwap_ratio = close[i] / vwap[i]
            if (0.995 <= price_vwap_ratio <= 1.005 and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price near VWAP (within 0.5%), downtrend, volume spike
            elif (0.995 <= price_vwap_ratio <= 1.005 and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below VWAP or trend fails
            if close[i] < vwap[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above VWAP or trend fails
            if close[i] > vwap[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals