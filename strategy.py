#!/usr/bin/env python3
"""
1h_Volume_Weighted_Trend_Follow
Hypothesis: In 1h timeframe, use 4h EMA(20) for trend direction and 1d VWAP for value area.
Long when price > 4h EMA(20) AND price > 1d VWAP with volume confirmation.
Short when price < 4h EMA(20) AND price < 1d VWAP with volume confirmation.
Volume confirmation requires volume > 1.5x 20-period average.
Designed for low trade frequency (15-30/year) with trend+value confirmation to work in both bull and bear markets.
"""

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
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 4h EMA(20) for trend
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d VWAP (typical price * volume) / volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = (typical_price_1d * volume_1d).cumsum() / volume_1d.cumsum()
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i]) or
            np.isnan(volume_conf[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema20 = ema_20_4h_aligned[i]
        vwap = vwap_1d_aligned[i]
        vol_confirm = volume_conf[i]
        
        if position == 0:
            # Long: price above both EMA and VWAP with volume confirmation
            if price > ema20 and price > vwap and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price below both EMA and VWAP with volume confirmation
            elif price < ema20 and price < vwap and vol_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: price falls below EMA OR VWAP
            if price < ema20 or price < vwap:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: price rises above EMA OR VWAP
            if price > ema20 or price > vwap:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Volume_Weighted_Trend_Follow"
timeframe = "1h"
leverage = 1.0