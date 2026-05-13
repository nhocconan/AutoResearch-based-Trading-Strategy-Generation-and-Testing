#!/usr/bin/env python3
"""
6h_WickReversal_Volume_Filter
Hypothesis: Long wicks (rejection candles) with volume confirmation at key levels 
work in both bull and bear markets. Long lower wick + volume spike = long signal 
(rejection of lower prices). Long upper wick + volume spike = short signal 
(rejection of higher prices). Uses 1w trend filter for higher timeframe bias.
Target: 20-50 trades/year per symbol.
"""

name = "6h_WickReversal_Volume_Filter"
timeframe = "6h"
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
    
    # Wick calculations
    body_size = np.abs(close - open_) if 'open' in prices.columns else np.abs(close - close)  # fallback
    # Actually calculate body size properly
    open_ = prices['open'].values
    body_size = np.abs(close - open_)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    
    # Long wick condition: wick > 2 * body size
    long_upper_wick = upper_wick > 2 * body_size
    long_lower_wick = lower_wick > 2 * body_size
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    # 1w trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    # Simple trend: price above/below 20-period EMA
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_1w = df_1w['close'].values > ema_20_1w
    downtrend_1w = df_1w['close'].values < ema_20_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Get values
        luw = long_upper_wick[i]
        llw = long_lower_wick[i]
        vol_conf = volume_conf[i]
        uptrend_htf = uptrend_1w_aligned[i]
        downtrend_htf = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: long lower wick + volume confirmation + 1w uptrend filter
            if llw and vol_conf and uptrend_htf:
                signals[i] = 0.25
                position = 1
            # SHORT: long upper wick + volume confirmation + 1w downtrend filter
            elif luw and vol_conf and downtrend_htf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: long upper wick appears (rejection of higher prices) or opposite signal
            if luw and vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: long lower wick appears (rejection of lower prices) or opposite signal
            if llw and vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals