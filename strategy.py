#!/usr/bin/env python3
"""
4h_1d_Price_Action_Reversal_V1
Hypothesis: Mean reversion at extreme daily levels with volume confirmation.
Uses 1-day ATR-based upper/lower bands as dynamic support/resistance.
Enters long when price touches lower band with volume spike in downtrend (RSI<40),
short when price touches upper band with volume spike in uptrend (RSI>60).
Works in both bull/bear by fading extremes only when momentum is exhausted.
Target: 20-35 trades/year via tight entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR-based bands
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR(10)
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align length
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(10, len(tr_1d)):
        atr_1d[i] = np.mean(tr_1d[i-9:i+1])
    
    # Dynamic bands: close ± 1.5 * ATR
    upper_band_1d = close_1d + 1.5 * atr_1d
    lower_band_1d = close_1d - 1.5 * atr_1d
    
    # Align bands to 4h
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band_1d)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band_1d)
    
    # 4h RSI(14) for momentum filter
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        avg_gain[i] = np.mean(gain[i-13:i+1])
        avg_loss[i] = np.mean(loss[i-13:i+1])
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14)  # volume MA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: touch lower band, volume spike, RSI < 40 (oversold)
            if (low[i] <= lower_band_aligned[i] and vol_spike[i] and rsi[i] < 40):
                signals[i] = 0.25
                position = 1
            # Short: touch upper band, volume spike, RSI > 60 (overbought)
            elif (high[i] >= upper_band_aligned[i] and vol_spike[i] and rsi[i] > 60):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Exit long: RSI > 50 (momentum shift) or price > upper band
            if (rsi[i] > 50 or high[i] >= upper_band_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 or price < lower band
            if (rsi[i] < 50 or low[i] <= lower_band_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Price_Action_Reversal_V1"
timeframe = "4h"
leverage = 1.0