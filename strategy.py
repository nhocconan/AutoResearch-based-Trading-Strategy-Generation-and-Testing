#!/usr/bin/env python3
"""
4h_1D_Camarilla_Breakout_MeanReversion_v1
Hypothesis: In bear markets, price reverses from extreme intraday deviations from daily VWAP.
Long when price < daily VWAP - 2*ATR and closes above VWAP. Short when price > daily VWAP + 2*ATR and closes below VWAP.
Uses daily VWAP and ATR for mean reversion signals. Works in both bull and bear by capturing mean reversion moves.
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
    
    # Daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Daily VWAP
    typical_price = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = np.cumsum(typical_price * vol_1d)
    vwap_denominator = np.cumsum(vol_1d)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    # Daily ATR (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align to 4h
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        vwap_val = vwap_aligned[i]
        atr_val = atr_aligned[i]
        
        # Mean reversion conditions
        long_signal = (close[i] < vwap_val - 2 * atr_val) and (close[i] > vwap_val)
        short_signal = (close[i] > vwap_val + 2 * atr_val) and (close[i] < vwap_val)
        
        # Exit conditions
        long_exit = close[i] < vwap_val
        short_exit = close[i] > vwap_val
        
        if position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
            elif short_signal:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1D_Camarilla_Breakout_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0