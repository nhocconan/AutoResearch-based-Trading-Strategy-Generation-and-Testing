#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use KAMA to determine trend direction, RSI(2) for mean-reversion entries, and Choppiness Index to filter ranging markets.
Go long when KAMA is rising (bullish trend) AND RSI(2) < 10 (oversold) AND Choppiness > 61.8 (ranging market).
Go short when KAMA is falling (bearish trend) AND RSI(2) > 90 (overbought) AND Choppiness > 61.8 (ranging market).
Exit when RSI(2) crosses back above 50 (long) or below 50 (short).
This strategy aims to capture mean-reversion moves within ranging markets while avoiding strong trends.
Designed to work in both bull and bear markets by focusing on range-bound conditions.
Target: 10-20 trades/year by combining strict entry filters.
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
    
    # Get 1d data (same as primary timeframe for indicators)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # KAMA(10) on 1d
    kama_period = 10
    fast_sc = 2 / (2 + 1)  # ER=10 -> fast EMA=2
    slow_sc = 2 / (30 + 1) # slow EMA=30
    
    kama_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= kama_period:
        kama_1d[kama_period-1] = close_1d[kama_period-1]  # seed
        for i in range(kama_period, len(close_1d)):
            # Efficiency Ratio
            change = abs(close_1d[i] - close_1d[i-kama_period])
            volatility = np.sum(np.abs(np.diff(close_1d[i-kama_period+1:i+1])))
            er = change / volatility if volatility != 0 else 0
            # Smoothing Constant
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            # KAMA
            kama_1d[i] = kama_1d[i-1] + sc * (close_1d[i] - kama_1d[i-1])
    
    # RSI(2) on 1d
    rsi_period = 2
    rsi_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= rsi_period + 1:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1d, np.nan)
        avg_loss = np.full_like(close_1d, np.nan)
        
        # First average
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1d = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14) on 1d
    chop_period = 14
    chop = np.full_like(close_1d, np.nan)
    if len(close_1d) >= chop_period:
        atr = np.zeros(len(close_1d))
        for i in range(1, len(close_1d)):
            tr = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]), 
                     abs(low_1d[i] - close_1d[i-1]))
            atr[i] = tr
        
        # Smooth ATR
        atr_ma = np.full_like(close_1d, np.nan)
        if len(atr) >= chop_period:
            atr_ma[chop_period-1] = np.mean(atr[1:chop_period])
            for i in range(chop_period, len(close_1d)):
                atr_ma[i] = (atr_ma[i-1] * (chop_period - 1) + atr[i]) / chop_period
        
        # Chop calculation
        for i in range(chop_period-1, len(close_1d)):
            highest_high = np.max(high_1d[i-chop_period+1:i+1])
            lowest_low = np.min(low_1d[i-chop_period+1:i+1])
            if atr_ma[i] != 0 and highest_high != lowest_low:
                chop[i] = 100 * np.log10(np.sum(atr[i-chop_period+1:i+1]) / (highest_high - lowest_low)) / np.log10(chop_period)
            else:
                chop[i] = 50  # neutral
    
    # Align indicators to 1d timeframe (already 1d, but using align for safety)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1w trend filter: EMA34
    ema34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False).values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kama_period, rsi_period+1, chop_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA slope (trend direction)
        kama_rising = kama_1d_aligned[i] > kama_1d_aligned[i-1]
        kama_falling = kama_1d_aligned[i] < kama_1d_aligned[i-1]
        
        if position == 0:
            # Long: KAMA rising (bullish) AND RSI(2) < 10 (oversold) AND Chop > 61.8 (ranging)
            if kama_rising and rsi_1d_aligned[i] < 10 and chop_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (bearish) AND RSI(2) > 90 (overbought) AND Chop > 61.8 (ranging)
            elif kama_falling and rsi_1d_aligned[i] > 90 and chop_aligned[i] > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI(2) crosses above 50
            if rsi_1d_aligned[i] > 50:
                signals[i] = -0.25  # close long
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI(2) crosses below 50
            if rsi_1d_aligned[i] < 50:
                signals[i] = 0.25  # close short
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0