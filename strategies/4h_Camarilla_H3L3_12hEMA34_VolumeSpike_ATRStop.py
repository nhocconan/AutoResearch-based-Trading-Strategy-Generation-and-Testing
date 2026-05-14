#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_12hEMA34_VolumeSpike_ATRStop
Hypothesis: Camarilla H3/L3 breakout on 4h with volume confirmation and 12h EMA trend filter.
Buy when price breaks above H3 with volume spike and uptrend; short when breaks below L3 with volume spike and downtrend.
Designed for 20-50 trades/year to avoid fee drag while capturing breakout moves in both bull and bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    typical = (high + low + close) / 3
    range_val = high - low
    H3 = close + range_val * 1.1 / 4
    L3 = close - range_val * 1.1 / 4
    return H3, L3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    
    # Camarilla H3/L3 on 4h
    H3, L3 = calculate_camarilla(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values)
    H3_4h = align_htf_to_ltf(prices, df_4h, H3)
    L3_4h = align_htf_to_ltf(prices, df_4h, L3)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(H3_4h[i]) or
            np.isnan(L3_4h[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema_34_12h_aligned[i]
        vol_spike = volume_spike[i]
        H3_val = H3_4h[i]
        L3_val = L3_4h[i]
        
        if position == 0:
            # Long: price breaks above H3 with volume spike and uptrend
            if not np.isnan(H3_val) and price > H3_val and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume spike and downtrend
            elif not np.isnan(L3_val) and price < L3_val and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below L3 OR trend turns down
            if not np.isnan(L3_val) and price < L3_val:
                signals[i] = 0.0
                position = 0
            elif price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above H3 OR trend turns up
            if not np.isnan(H3_val) and price > H3_val:
                signals[i] = 0.0
                position = 0
            elif price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0