#!/usr/bin/env python3
# 4h_KAMA_RSI_ChopFilter
# Hypothesis: 4-hour KAMA trend direction combined with RSI mean reversion and Choppiness index regime filter.
# KAMA adapts to market noise, reducing whipsaws in ranging markets. RSI identifies overbought/oversold conditions
# for mean reversion entries. Choppiness index filters trades to only occur in ranging markets (CHOP > 61.8)
# where mean reversion is most effective, avoiding strong trends where RSI fails. Works in both bull and bear
# markets by adapting to regime conditions. Target: 20-40 trades per year (~80-160 over 4 years) with position size 0.25.

name = "4h_KAMA_RSI_ChopFilter"
timeframe = "4h"
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
    
    # Load 1d data ONCE for Choppiness index (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness index on daily data
    def calculate_choppiness(high_arr, low_arr, close_arr, period=14):
        atr = []
        for i in range(len(close_arr)):
            if i == 0:
                tr = high_arr[i] - low_arr[i]
            else:
                tr = max(high_arr[i] - low_arr[i], abs(high_arr[i] - close_arr[i-1]), abs(low_arr[i] - close_arr[i-1]))
            atr.append(tr)
        
        atr_arr = np.array(atr)
        if len(atr_arr) < period:
            return np.full_like(close_arr, 50.0, dtype=float)
        
        # Sum of ATR over period
        atr_sum = np.convolve(atr_arr, np.ones(period), 'valid')
        # True range over period (max high - min low)
        hh = np.array([np.max(high_arr[i:i+period]) if i+period <= len(high_arr) else np.nan 
                      for i in range(len(high_arr))])
        ll = np.array([np.min(low_arr[i:i+period]) if i+period <= len(low_arr) else np.nan 
                      for i in range(len(low_arr))])
        range_period = hh - ll
        
        chop = np.full_like(close_arr, 50.0, dtype=float)
        valid_idx = ~np.isnan(range_period) & (range_period > 0)
        chop[valid_idx] = 100 * np.log10(atr_sum[valid_idx] / range_period[valid_idx]) / np.log10(period)
        return chop
    
    chop = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # KAMA (Kaufman Adaptive Moving Average) for trend
    def kama(close, er_len=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 0 else np.array([])
        # Handle array operations properly
        er = np.zeros_like(close)
        for i in range(er_len, len(close)):
            if volatility[i-er_len:i] > 0:
                er[i] = change[i-er_len] / np.sum(np.abs(np.diff(close[i-er_len:i+1])))
            else:
                er[i] = 0
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_val = kama(close)
    
    # RSI for mean reversion
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
        
        # Wilder smoothing
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = np.zeros_like(close)
        rsi_out = 100 - (100 / (1 + rs))
        # Set first period values to 50 (neutral)
        rsi_out[:period] = 50
        return rsi_out
    
    rsi_val = rsi(close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA trend: price above/below KAMA
        price_above_kama = close[i] > kama_val[i]
        price_below_kama = close[i] < kama_val[i]
        
        # RSI conditions for mean reversion
        rsi_overbought = rsi_val[i] > 70
        rsi_oversold = rsi_val[i] < 30
        
        # Choppiness filter: only trade in ranging markets (CHOP > 61.8)
        ranging_market = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price below KAMA (dip) + RSI oversold + ranging market
            if price_below_kama and rsi_oversold and ranging_market:
                signals[i] = 0.25
                position = 1
            # Short: price above KAMA (bounce) + RSI overbought + ranging market
            elif price_above_kama and rsi_overbought and ranging_market:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses above KAMA or RSI overbought
            if price_above_kama or rsi_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses below KAMA or RSI oversold
            if price_below_kama or rsi_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals