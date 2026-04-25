#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) identifies trend direction,
RSI(14) filters overbought/oversold extremes, and Choppiness Index (CHOP) regime
filter avoids whipsaws in range markets. Works in bull/bear via KAMA trend.
Discrete sizing (0.25) targets 50-100 trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and chop regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter
    def calculate_kama(close, length=10, fast=2, slow=30):
        close_s = pd.Series(close)
        change = np.abs(close_s.diff(length))
        volatility = close_s.diff().abs().rolling(length, min_periods=1).sum()
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    close_1w = df_1w['close'].values
    kama_1w = calculate_kama(close_1w, length=10, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate weekly Choppiness Index for regime filter
    def calculate_chop(high, low, close, length=14):
        high_s, low_s, close_s = pd.Series(high), pd.Series(low), pd.Series(close)
        atr = np.maximum(high_s - low_s, 
                         np.maximum(np.abs(high_s - close_s.shift(1)), 
                                    np.abs(low_s - close_s.shift(1))))
        atr_sum = atr.rolling(length, min_periods=length).sum()
        highest_high = high_s.rolling(length, min_periods=length).max()
        lowest_low = low_s.rolling(length, min_periods=length).min()
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(length)
        return chop.values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    chop_1w = calculate_chop(high_1w, low_1w, close_1w, length=14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate daily RSI for entry timing
    def calculate_rsi(close, length=14):
        close_s = pd.Series(close)
        delta = close_s.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    rsi_1d = calculate_rsi(close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(rsi_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        kama_trend = kama_1w_aligned[i]
        chop_value = chop_1w_aligned[i]
        rsi_value = rsi_1d[i]
        
        # Regime filter: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
        # We'll use trend-following logic when CHOP < 50 (more permissive to capture trends)
        trending_regime = chop_value < 50
        
        # Exit conditions: opposite signal or regime change to range
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit long: price below KAMA OR RSI overbought OR regime becomes range
                if curr_close < kama_trend or rsi_value > 70 or not trending_regime:
                    exit_signal = True
            elif position == -1:
                # Exit short: price above KAMA OR RSI oversold OR regime becomes range
                if curr_close > kama_trend or rsi_value < 30 or not trending_regime:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: KAMA alignment + RSI extreme + trending regime
        if position == 0:
            # Long: price above KAMA AND RSI oversold AND trending regime
            long_condition = (curr_close > kama_trend) and (rsi_value < 30) and trending_regime
            # Short: price below KAMA AND RSI overbought AND trending regime
            short_condition = (curr_close < kama_trend) and (rsi_value > 70) and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopRegime_v1"
timeframe = "1d"
leverage = 1.0