#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d KAMA trend direction with RSI mean reversion and choppiness filter
# Long when 1d KAMA is rising AND 1h RSI < 30 (oversold) AND 1d choppiness > 61.8 (range regime)
# Short when 1d KAMA is falling AND 1h RSI > 70 (overbought) AND 1d choppiness > 61.8 (range regime)
# Exit when 1h RSI crosses 50 (mean reversion completion)
# Uses discrete sizing 0.25 to limit drawdown and reduce fee churn
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# KAMA adapts to market noise, RSI provides mean reversion timing, chop filter ensures ranging markets

name = "12h_1dKAMA_Trend_1hRSI_MR_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for KAMA trend and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d KAMA (adaptive moving average)
    def calculate_kama(close, period=10, fast=2, slow=30):
        # Efficiency ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else np.array([0])
        # Handle array operations properly
        er = np.zeros_like(close)
        for i in range(period, len(close)):
            if np.sum(np.abs(np.diff(close[i-period:i+1]))) > 0:
                er[i] = np.abs(close[i] - close[i-period]) / np.sum(np.abs(np.diff(close[i-period:i+1])))
            else:
                er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA calculation
        kama = np.full_like(close, np.nan, dtype=float)
        kama[period] = close[period]  # seed
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1d = calculate_kama(close_1d, period=10, fast=2, slow=30)
    kama_rising_1d = np.zeros_like(kama_1d, dtype=bool)
    kama_falling_1d = np.zeros_like(kama_1d, dtype=bool)
    for i in range(1, len(kama_1d)):
        if not np.isnan(kama_1d[i]) and not np.isnan(kama_1d[i-1]):
            kama_rising_1d[i] = kama_1d[i] > kama_1d[i-1]
            kama_falling_1d[i] = kama_1d[i] < kama_1d[i-1]
    
    # Calculate 1d RSI for additional confirmation
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close, np.nan, dtype=float)
        avg_loss = np.full_like(close, np.nan, dtype=float)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, period=14)
    
    # Calculate 1d Choppiness Index
    def calculate_chop(high, low, close, period=14):
        # True range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Sum of true range over period
        atr_sum = np.full_like(close, np.nan, dtype=float)
        for i in range(period-1, len(tr)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_h = np.full_like(close, np.nan, dtype=float)
        min_l = np.full_like(close, np.nan, dtype=float)
        for i in range(period-1, len(high)):
            max_h[i] = np.max(high[i-period+1:i+1])
            min_l[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.full_like(close, np.nan, dtype=float)
        for i in range(period-1, len(close)):
            if atr_sum[i] > 0 and (max_h[i] - min_l[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_h[i] - min_l[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, period=14)
    chop_high_regime_1d = chop_1d > 61.8  # ranging market
    
    # Get 1h data ONCE before loop for RSI entry timing
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    close_1h = df_1h['close'].values
    
    # Calculate 1h RSI
    rsi_1h = calculate_rsi(close_1h, period=14)
    rsi_oversold_1h = rsi_1h < 30
    rsi_overbought_1h = rsi_1h > 70
    rsi_exit_1h = (rsi_1h > 50) & (rsi_1h < 70)  # exit long when RSI crosses 50 from below
    rsi_exit_short_1h = (rsi_1h < 50) & (rsi_1h > 30)  # exit short when RSI crosses 50 from above
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    kama_rising_aligned = align_htf_to_ltf(prices, df_1d, kama_rising_1d)
    kama_falling_aligned = align_htf_to_ltf(prices, df_1d, kama_falling_1d)
    chop_high_aligned = align_htf_to_ltf(prices, df_1d, chop_high_regime_1d)
    
    # Align 1h indicators to 12h timeframe (wait for completed 1h bar)
    rsi_oversold_aligned = align_htf_to_ltf(prices, df_1h, rsi_oversold_1h)
    rsi_overbought_aligned = align_htf_to_ltf(prices, df_1h, rsi_overbought_1h)
    rsi_exit_aligned = align_htf_to_ltf(prices, df_1h, rsi_exit_1h)
    rsi_exit_short_aligned = align_htf_to_ltf(prices, df_1h, rsi_exit_short_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(kama_rising_aligned[i]) or np.isnan(kama_falling_aligned[i]) or 
            np.isnan(chop_high_aligned[i]) or np.isnan(rsi_oversold_aligned[i]) or 
            np.isnan(rsi_overbought_aligned[i]) or np.isnan(rsi_exit_aligned[i]) or 
            np.isnan(rsi_exit_short_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising AND RSI oversold AND choppy market
            if (kama_rising_aligned[i] and rsi_oversold_aligned[i] and chop_high_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND RSI overbought AND choppy market
            elif (kama_falling_aligned[i] and rsi_overbought_aligned[i] and chop_high_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 50 (mean reversion completion)
            if rsi_exit_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI crosses below 50 (mean reversion completion)
            if rsi_exit_short_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals