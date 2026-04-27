#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Mean Reversion with 4h Trend Filter and Volume Confirmation
# In range-bound markets, price reverts to the 4h VWAP with high probability.
# Uses 4h VWAP as mean reversion target and volume spike for entry confirmation.
# Designed to work in both bull and bear markets by fading extremes.
# Target: 20-40 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for VWAP calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate VWAP on 4h data
    typical_price_4h = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3
    vwap_4h = np.full(len(df_4h), np.nan)
    cum_vol = 0.0
    cum_price_vol = 0.0
    for i in range(len(df_4h)):
        tp = typical_price_4h[i]
        vol = df_4h['volume'].values[i]
        cum_price_vol += tp * vol
        cum_vol += vol
        if cum_vol > 0:
            vwap_4h[i] = cum_price_vol / cum_vol
    
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # Calculate ATR for volatility filter
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_4h_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Deviation from 4h VWAP in ATR units
        deviation = (close[i] - vwap_4h_aligned[i]) / atr[i]
        
        if position == 0:
            # Long entry: price below VWAP + volume spike
            if deviation < -1.5 and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: price above VWAP + volume spike
            elif deviation > 1.5 and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to VWAP or opposite extreme
            if deviation > -0.5 or deviation > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to VWAP or opposite extreme
            if deviation < 0.5 or deviation < -1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_MeanReversion_4hVWAP_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0