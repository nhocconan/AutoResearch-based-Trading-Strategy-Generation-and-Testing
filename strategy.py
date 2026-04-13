#!/usr/bin/env python3
"""
4h_1d_RSI_MeanReversion_With_Volume_Filter
Hypothesis: Mean reversion works in range-bound markets. Enter long when RSI(14) < 30 and price > daily VWAP,
short when RSI(14) > 70 and price < daily VWAP, with volume confirmation (>1.5x 20-day average volume).
Exit when RSI crosses back above 50 (long) or below 50 (short).
Uses daily timeframe for RSI/VWAP/volume to reduce trade frequency and avoid whipsaw.
Designed to work in both bull (buy dips) and bear (sell rallies) markets by fading extremes.
Target: 20-40 trades/year per symbol for low friction and high edge.
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
    
    # Daily data for RSI, VWAP, volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # RSI(14) calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily VWAP calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = np.cumsum(typical_price * vol_1d)
    vwap_denominator = np.cumsum(vol_1d)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    # Volume MA(20)
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    
    # Align 1d data to 4h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20.values)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(vwap_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1d volume > 1.5x 20-period average
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 1.5)
        
        # Mean reversion conditions
        long_condition = (rsi_aligned[i] < 30) and (close[i] > vwap_aligned[i]) and vol_condition
        short_condition = (rsi_aligned[i] > 70) and (close[i] < vwap_aligned[i]) and vol_condition
        
        # Exit conditions: RSI crosses 50
        long_exit = rsi_aligned[i] > 50
        short_exit = rsi_aligned[i] < 50
        
        if position == 0:
            if long_condition:
                position = 1
                signals[i] = position_size
            elif short_condition:
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

name = "4h_1d_RSI_MeanReversion_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0