#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v1
Hypothesis: KAMA adapts to market noise, providing reliable trend signals in both trending and ranging markets.
Combined with RSI momentum and Choppiness Index regime filter to avoid false signals.
Only takes positions when KAMA confirms trend, RSI shows momentum, and market is not too choppy.
Designed for low trade frequency (<20 trades/year) to minimize fee impact while capturing major trends.
Works in bull markets via trend following and in bear markets via inverse signals during weak trends.
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
    
    # Get weekly data for trend filter and chop filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
        kama = np.full_like(close, np.nan)
        if len(close) < er_period:
            return kama
        
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[er_period:] = change[er_period-1:] / np.maximum(volatility[er_period-1:], 1e-10)
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # Initialize KAMA
        kama[er_period] = close[er_period]
        
        # Calculate KAMA
        for i in range(er_period + 1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        
        return kama
    
    # Calculate RSI
    def calculate_rsi(close, period=14):
        rsi = np.full_like(close, np.nan)
        if len(close) < period + 1:
            return rsi
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.zeros_like(close)
        rs[period:] = avg_gain[period:] / np.maximum(avg_loss[period:], 1e-10)
        rsi[period:] = 100 - (100 / (1 + rs[period:]))
        
        return rsi
    
    # Calculate Choppiness Index
    def calculate_choppiness(high, low, close, period=14):
        chop = np.full_like(close, np.nan)
        if len(close) < period:
            return chop
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR (average true range)
        atr = np.zeros_like(tr)
        for i in range(1, len(tr)):
            if i < period:
                atr[i] = np.mean(tr[1:i+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Sum of ATR over period
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                max_high[i] = np.max(high[:i+1])
                min_low[i] = np.min(low[:i+1])
            else:
                max_high[i] = np.max(high[i-period+1:i+1])
                min_low[i] = np.min(low[i-period+1:i+1])
        
        # Choppiness Index
        for i in range(period, len(close)):
            if atr_sum[i] > 0 and max_high[i] > min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50.0
        
        return chop
    
    # Calculate indicators on weekly data
    kama_1w = calculate_kama(close_1w, er_period=10, fast_sc=2, slow_sc=30)
    rsi_1w = calculate_rsi(close_1w, period=14)
    chop_1w = calculate_choppiness(high_1w, low_1w, close_1w, period=14)
    
    # Align to daily timeframe
    kama = align_htf_to_ltf(prices, df_1w, kama_1w)
    rsi = align_htf_to_ltf(prices, df_1w, rsi_1w)
    chop = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure we have enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        rsi_overbought = rsi[i] > 60
        rsi_oversold = rsi[i] < 40
        not_choppy = chop[i] < 61.8  # Not in choppy regime
        
        if position == 0:
            # Long: price above KAMA, RSI not overbought, not choppy
            if price_above_kama and not rsi_overbought and not_choppy:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI not oversold, not choppy
            elif price_below_kama and not rsi_oversold and not_choppy:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA OR RSI overbought OR market becomes choppy
            if (price_below_kama or rsi[i] > 70 or chop[i] > 61.8):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA OR RSI oversold OR market becomes choppy
            if (price_above_kama or rsi[i] < 30 or chop[i] > 61.8):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0