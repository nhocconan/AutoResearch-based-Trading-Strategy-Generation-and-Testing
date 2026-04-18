#!/usr/bin/env python3
"""
12h_VWAP_Breakout_1wTrend_Filter
Hypothesis: Price breaking above/below the 12h VWAP with weekly trend alignment (price > weekly EMA34) and volume confirmation captures strong momentum. Exit on VWAP reversion or trend weakening. Designed for low trade frequency to avoid fee drag while capturing strong trending moves in both bull and bear markets.
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
    
    # VWAP calculation for 12h
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Need warmup for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(vwap[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        ema_34 = ema_34_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price > VWAP, price > weekly EMA34, volume confirmation
            if price > vwap_val and price > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price < VWAP, price < weekly EMA34, volume confirmation
            elif price < vwap_val and price < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < VWAP OR price < weekly EMA34
            if price < vwap_val or price < ema_34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > VWAP OR price > weekly EMA34
            if price > vwap_val or price > ema_34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_VWAP_Breakout_1wTrend_Filter"
timeframe = "12h"
leverage = 1.0