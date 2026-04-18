#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_1wEMA34_Volume_Filter
Hypothesis: Keltner Channel breakouts on 1d timeframe with 1-week EMA34 trend filter and volume confirmation. 
Enters long when price breaks above upper KC with bullish weekly trend and volume spike, short when breaks below lower KC with bearish weekly trend and volume spike.
Designed for low trade frequency (~10-20/year) with trend-following capability in both bull and bear markets.
Weekly trend filter reduces whipsaws and improves performance during bear markets like 2022.
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
    
    # Keltner Channel (20, 2.0) on daily
    atr_period = 20
    atr = np.full(n, np.nan)
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    
    for i in range(atr_period, n):
        if i == atr_period:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    ma = np.full(n, np.nan)
    for i in range(20, n):
        if i == 20:
            ma[i] = np.mean(close[0:20])
        else:
            ma[i] = (ma[i-1] * 19 + close[i]) / 20
    
    kc_upper = ma + (2 * atr)
    kc_lower = ma - (2 * atr)
    
    # Weekly EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    k = 2 / (34 + 1)
    for i in range(34, len(close_1w)):
        if i == 34:
            ema34_1w[i] = np.mean(close_1w[0:35])
        else:
            ema34_1w[i] = close_1w[i] * k + ema34_1w[i-1] * (1 - k)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper KC with bullish weekly trend and volume spike
            if close[i] > kc_upper[i] and ema34_1w_aligned[i] > close_1w[-1] if len(close_1w) > 0 else False and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower KC with bearish weekly trend and volume spike
            elif close[i] < kc_lower[i] and ema34_1w_aligned[i] < close_1w[-1] if len(close_1w) > 0 else False and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below lower KC or weekly trend turns bearish
            if close[i] < kc_lower[i] or ema34_1w_aligned[i] < close_1w[-1] if len(close_1w) > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above upper KC or weekly trend turns bullish
            if close[i] > kc_upper[i] or ema34_1w_aligned[i] > close_1w[-1] if len(close_1w) > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Keltner_Channel_Breakout_1wEMA34_Volume_Filter"
timeframe = "1d"
leverage = 1.0