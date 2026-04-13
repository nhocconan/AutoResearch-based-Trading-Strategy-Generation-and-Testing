#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA trend + RSI + choppiness regime filter
    # Long: KAMA rising AND RSI > 50 AND CHOP < 45 (trending market)
    # Short: KAMA falling AND RSI < 50 AND CHOP < 45 (trending market)
    # Exit: KAMA direction reverses OR CHOP > 55 (choppy/ranging market)
    # Uses 1d for all indicators to match primary timeframe
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 30-80 total trades over 4 years (~7-20/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for all indicators (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array alignment
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # start with first close
    for i in range(10, len(close_1d)):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.diff(kama, prepend=kama[0])
    kama_dir = np.where(kama_dir > 0, 1, np.where(kama_dir < 0, -1, 0))
    
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP) over 14 periods
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14) - sum of TR over 14 periods
    atr_sum = np.zeros_like(tr)
    for i in range(14, len(tr)):
        atr_sum[i] = np.sum(tr[i-13:i+1])  # sum of last 14 TR values
    
    # Highest high and lowest low over 14 periods
    max_high = np.zeros_like(high_1d)
    min_low = np.zeros_like(low_1d)
    for i in range(14, len(high_1d)):
        max_high[i] = np.max(high_1d[i-13:i+1])
        min_low[i] = np.min(low_1d[i-13:i+1])
    
    # CHOP = 100 * log10(atr_sum / (max_high - min_low)) / log10(14)
    range_hl = max_high - min_low
    chop = np.full_like(close_1d, np.nan)
    mask = (range_hl > 0) & (~np.isnan(atr_sum)) & (~np.isnan(range_hl))
    chop[mask] = 100 * np.log10(atr_sum[mask] / range_hl[mask]) / np.log10(14)
    
    # Align all 1d indicators to 1d timeframe (no additional delay for price-based indicators)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(kama_dir_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when CHOP < 45 (trending market)
        trending_market = chop_aligned[i] < 45
        # Exit regime: CHOP > 55 (choppy/ranging market)
        choppy_market = chop_aligned[i] > 55
        
        # KAMA direction signals
        kama_rising = kama_dir_aligned[i] > 0
        kama_falling = kama_dir_aligned[i] < 0
        
        # RSI filter: RSI > 50 for bullish bias, RSI < 50 for bearish bias
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Entry logic: KAMA direction + RSI filter + trending regime
        long_entry = kama_rising and rsi_bullish and trending_market
        short_entry = kama_falling and rsi_bearish and trending_market
        
        # Exit logic: KAMA direction reverses OR regime shifts to choppy
        long_exit = (kama_dir_aligned[i] <= 0) or choppy_market
        short_exit = (kama_dir_aligned[i] >= 0) or choppy_market
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "1d_kama_rsi_chop_regime_v1"
timeframe = "1d"
leverage = 1.0