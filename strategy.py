#!/usr/bin/env python3
"""
6h_1w_VWAP_Deviation_Mean_Reversion_v1
Hypothesis: In weekly mean-reverting markets, price deviates from weekly VWAP but reverts.
Long when price < weekly VWAP - 1.5*weekly ATR and weekly RSI < 40.
Short when price > weekly VWAP + 1.5*weekly ATR and weekly RSI > 60.
Exit when price crosses weekly VWAP.
Designed for 6h timeframe to capture mean reversion in both bull and bear markets with low trade frequency.
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
    
    # Weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    vol_1w = df_1w['volume'].values
    
    # Weekly VWAP calculation
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    vwap_numerator = np.cumsum(typical_price_1w * vol_1w)
    vwap_denominator = np.cumsum(vol_1w)
    vwap_1w = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price_1w)
    
    # Weekly ATR (14-period)
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Weekly RSI (14-period)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align weekly data to 6h
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or
            np.isnan(rsi_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        lower_band = vwap_1w_aligned[i] - 1.5 * atr_1w_aligned[i]
        upper_band = vwap_1w_aligned[i] + 1.5 * atr_1w_aligned[i]
        
        long_entry = close[i] < lower_band and rsi_1w_aligned[i] < 40
        short_entry = close[i] > upper_band and rsi_1w_aligned[i] > 60
        
        # Exit condition: price crosses weekly VWAP
        long_exit = close[i] > vwap_1w_aligned[i]
        short_exit = close[i] < vwap_1w_aligned[i]
        
        if position == 0:
            if long_entry:
                position = 1
                signals[i] = position_size
            elif short_entry:
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

name = "6h_1w_VWAP_Deviation_Mean_Reversion_v1"
timeframe = "6h"
leverage = 1.0