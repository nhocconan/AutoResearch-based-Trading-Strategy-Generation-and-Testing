#!/usr/bin/env python3
# 6h_12h_1d_volume_weighted_momentum_v1
# Hypothesis: 6-hour momentum filtered by 12h/1d volume-weighted average price (VWAP) and volume confirmation.
# Long: price > 6h momentum > 0 AND price > 12h VWAP AND volume > 1.3x 20-period average volume.
# Short: price < 6h momentum < 0 AND price < 12h VWAP AND volume > 1.3x 20-period average volume.
# Exit: momentum reverses or price crosses 12h VWAP.
# Designed to capture institutional flow in both bull and bear markets with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_volume_weighted_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6-hour momentum (10-period ROC)
    momentum = np.full(n, np.nan)
    for i in range(10, n):
        momentum[i] = (close[i] - close[i-10]) / close[i-10]
    
    # 20-period average volume
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 12-hour VWAP (typical price * volume)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    typical_price_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3
    vwap_12h = np.full(len(typical_price_12h), np.nan)
    cum_vol_price = 0.0
    cum_vol = 0.0
    
    for i in range(len(typical_price_12h)):
        pv = typical_price_12h[i] * df_12h['volume'].values[i]
        vol = df_12h['volume'].values[i]
        cum_vol_price += pv
        cum_vol += vol
        if cum_vol > 0:
            vwap_12h[i] = cum_vol_price / cum_vol
    
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        mom = momentum[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        vwap = vwap_12h_aligned[i]
        price = close[i]
        
        if np.isnan(mom) or np.isnan(avg_vol) or np.isnan(vwap):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.3 * avg_vol
        
        if position == 1:  # Long position
            if mom <= 0 or price < vwap:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if mom >= 0 or price > vwap:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if mom > 0 and price > vwap and vol_surge:
                position = 1
                signals[i] = 0.25
            elif mom < 0 and price < vwap and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals