#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Chop_Filter_v1
Hypothesis: Use KAMA to determine trend direction on 4h, combine with RSI overbought/oversold and chop filter to avoid sideways markets. Enter long when KAMA upward + RSI < 30 + chop > 61.8, short when KAMA downward + RSI > 70 + chop > 61.8. This targets mean-reversion within trending markets, works in bull/bear via adaptive trend filter. Low trade frequency expected due to strict triple condition.
"""

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
    
    # Get daily data for chop filter
    df_1d = get_htf_data(prices, '1d')
    
    # 4h data for KAMA and RSI
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate KAMA on 4h close
    close_4h = df_4h['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_4h, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_4h)), axis=0)  # placeholder - will compute properly
    # Proper ER calculation
    er = np.zeros_like(close_4h)
    for i in range(10, len(close_4h)):
        direction = np.abs(close_4h[i] - close_4h[i-10])
        volatility_sum = np.sum(np.abs(np.diff(close_4h[i-9:i+1])))
        if volatility_sum > 0:
            er[i] = direction / volatility_sum
        else:
            er[i] = 0
    er[0:10] = 0
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # KAMA
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # RSI on 4h close
    delta = np.diff(close_4h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, 50), rsi])  # pad beginning
    
    # Chop index on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_period = 14
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(np.roll(close_1d, 1) - low_1d)))
    tr[0] = high_1d[0] - low_1d[0]
    atr = np.zeros_like(tr)
    for i in range(1, len(tr)):
        if i < atr_period:
            atr[i] = np.mean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    # Chop calculation
    sum_tr = np.zeros_like(atr)
    max_h = np.zeros_like(atr)
    min_l = np.zeros_like(atr)
    for i in range(atr_period, len(tr)):
        sum_tr[i] = np.sum(tr[i-atr_period+1:i+1])
        max_h[i] = np.max(high_1d[i-atr_period+1:i+1])
        min_l[i] = np.min(low_1d[i-atr_period+1:i+1])
    chop = np.where((max_h - min_l) != 0, 100 * np.log10(sum_tr / (max_h - min_l)) / np.log10(atr_period), 50)
    chop = np.concatenate([np.full(atr_period, 50), chop[atr_period:]])  # pad beginning
    
    # Align all higher timeframe data to 4h
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: slope of KAMA
        kama_slope = kama_4h_aligned[i] - kama_4h_aligned[i-1] if i > 0 else 0
        
        if position == 0:
            # Long: KAMA upward + RSI oversold + choppy market (mean reversion opportunity)
            if kama_slope > 0 and rsi_4h_aligned[i] < 30 and chop_1d_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA downward + RSI overbought + choppy market
            elif kama_slope < 0 and rsi_4h_aligned[i] > 70 and chop_1d_aligned[i] > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down or RSI overbought or chop decreases (trending)
            if kama_slope <= 0 or rsi_4h_aligned[i] > 70 or chop_1d_aligned[i] < 50:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up or RSI oversold or chop decreases
            if kama_slope >= 0 or rsi_4h_aligned[i] < 30 or chop_1d_aligned[i] < 50:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_RSI_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0