# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_kama_rsi_chop
# Uses 12h KAMA to determine trend direction, combined with 12h RSI for momentum,
# and 1d Choppiness Index to filter ranging markets.
# Long when KAMA > previous KAMA (up trend), RSI < 50 (pullback in uptrend), and CHOP > 61.8 (ranging market).
# Short when KAMA < previous KAMA (down trend), RSI > 50 (pullback in downtrend), and CHOP > 61.8 (ranging market).
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
# Works in ranging markets by buying pullbacks in the trend direction within chop regimes.
# Focus on BTC/ETH as primary targets.

name = "12h_1d_kama_rsi_chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value is NaN
    
    # ATR (14-period smoothed)
    atr = np.zeros_like(tr)
    atr[0] = np.nan
    for i in range(1, len(tr)):
        if np.isnan(atr[i-1]) or np.isnan(tr[i]):
            atr[i] = np.nan
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    atr_sum = np.zeros_like(atr)
    for i in range(len(atr_sum)):
        if i < 13:
            atr_sum[i] = np.nan
        else:
            atr_sum[i] = np.nansum(atr[i-13:i+1])
    
    # High-Low range over 14 periods
    max_high = np.zeros_like(high_1d)
    min_low = np.zeros_like(low_1d)
    for i in range(len(max_high)):
        if i < 13:
            max_high[i] = np.nan
            min_low[i] = np.nan
        else:
            max_high[i] = np.nanmax(high_1d[i-13:i+1])
            min_low[i] = np.nanmin(low_1d[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    for i in range(len(chop)):
        if np.isnan(atr_sum[i]) or np.isnan(max_high[i]) or np.isnan(min_low[i]) or max_high[i] == min_low[i]:
            chop[i] = np.nan
        else:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
    
    # Align daily Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h KAMA (10-period ER, 2 and 30 SC)
    # Efficiency Ratio
    change = np.zeros_like(close)
    for i in range(len(change)):
        if i == 0:
            change[i] = 0
        else:
            change[i] = abs(close[i] - close[i-1])
    
    # Sum of absolute changes over 10 periods
    abs_change_sum = np.zeros_like(change)
    for i in range(len(abs_change_sum)):
        if i < 9:
            abs_change_sum[i] = np.nan
        else:
            abs_change_sum[i] = np.nansum(change[i-9:i+1])
    
    # Net change over 10 periods
    net_change = np.abs(np.subtract(close[9:], close[:-9]))
    net_change_padded = np.full_like(close, np.nan)
    net_change_padded[9:] = net_change
    
    # Efficiency Ratio
    er = np.zeros_like(close)
    for i in range(len(er)):
        if np.isnan(abs_change_sum[i]) or abs_change_sum[i] == 0:
            er[i] = 0
        else:
            er[i] = net_change_padded[i] / abs_change_sum[i]
    
    # Smoothing Constants
    sc_fast = 2 / (2 + 1)   # for EMA 2
    sc_slow = 2 / (30 + 1)  # for EMA 30
    sc = np.zeros_like(er)
    for i in range(len(sc)):
        sc[i] = (er[i] * (sc_fast - sc_slow) + sc_slow) ** 2
    
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(kama)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 12h RSI (14-period)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Average gain and loss
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(len(avg_gain)):
        if i < 13:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        else:
            if i == 13:
                avg_gain[i] = np.nansum(gain[0:14]) / 14
                avg_loss[i] = np.nansum(loss[0:14]) / 14
            else:
                if np.isnan(avg_gain[i-1]) or np.isnan(avg_loss[i-1]):
                    avg_gain[i] = np.nan
                    avg_loss[i] = np.nan
                else:
                    avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                    avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    # RSI
    rsi = np.zeros_like(close)
    for i in range(len(rsi)):
        if np.isnan(avg_gain[i]) or np.isnan(avg_loss[i]) or avg_loss[i] == 0:
            rsi[i] = 50  # neutral when undefined
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(chop_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade in ranging markets (CHOP > 61.8)
        if chop_aligned[i] <= 61.8:
            # Hold current position if chop filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: KAMA up trend + RSI pullback (< 50)
        if kama[i] > kama[i-1] and rsi[i] < 50 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: KAMA down trend + RSI pullback (> 50)
        elif kama[i] < kama[i-1] and rsi[i] > 50 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit when trend changes
        elif position == 1 and kama[i] <= kama[i-1]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and kama[i] >= kama[i-1]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals