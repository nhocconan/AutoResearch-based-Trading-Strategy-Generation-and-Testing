#!/usr/bin/env python3
"""
1d KAMA Direction + RSI + Chop Filter
Uses KAMA (Kaufman Adaptive Moving Average) to capture trend direction,
combined with RSI for overbought/oversold conditions and Choppiness Index
to filter for trending markets. Designed for low trade frequency with
strong edge in both bull and bear markets by trading with the trend in
trending conditions and avoiding chop.
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
    
    # Get 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d
    # Efficiency Ratio: ER = |Close - Close[10]| / Sum(|Close - Close[1]|) over 10 periods
    change = np.abs(close_1d[10:] - close_1d[:-10])
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # temporary fix, will compute properly below
    
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        direction = np.abs(close_1d[i] - close_1d[i-10])
        volatility_sum = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
        if volatility_sum > 0:
            er[i] = direction / volatility_sum
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[9] = close_1d[9]  # start with close
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Get 1w data for trend filter (optional, but we can use it for extra confirmation)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # RSI calculation (14-period)
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        # First average
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period+1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close_1d, 14)
    
    # Choppiness Index calculation (14-period)
    def calculate_choppiness(high_prices, low_prices, close_prices, period=14):
        # True Range
        tr1 = high_prices[1:] - low_prices[1:]
        tr2 = np.abs(high_prices[1:] - close_prices[:-1])
        tr3 = np.abs(low_prices[1:] - close_prices[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.max(tr[:3]) if len(tr) >= 3 else 0], tr])  # align length
        
        # Sum of True Range over period
        atr_sum = np.zeros_like(close_prices)
        for i in range(period, len(close_prices)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close_prices)
        lowest_low = np.zeros_like(close_prices)
        for i in range(period-1, len(close_prices)):
            highest_high[i] = np.max(high_prices[i-period+1:i+1])
            lowest_low[i] = np.min(low_prices[i-period+1:i+1])
        
        # Chop = 100 * log10(ATRsum / (HH - LL)) / log10(period)
        hh_ll = highest_high - lowest_low
        chop = np.zeros_like(close_prices)
        for i in range(period, len(close_prices)):
            if hh_ll[i] > 0 and atr_sum[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / hh_ll[i]) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, close_1d, 14)
    
    # Align all indicators to lower timeframe (1d is already our base, but we align for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 14, 10)  # enough for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        
        # Only trade in trending markets (Chop < 61.8)
        if chop_val > 61.8:
            # In chop, go flat
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA and RSI not overbought
            if price > kama_val and rsi_val < 70:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and RSI not oversold
            elif price < kama_val and rsi_val > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price crosses below KAMA or RSI overbought
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price crosses above KAMA or RSI oversold
            if price > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0