#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop_filter_v1
Hypothesis: Daily strategy using KAMA trend direction, RSI for momentum/mean-reversion, and Choppiness Index for regime filtering.
Only takes trades when KAMA trend aligns with higher timeframe (weekly) trend, RSI is in extreme territory, and market is trending (not choppy).
Designed to work in both bull and bear markets by following the weekly trend and using RSI extremes for entry.
Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Kaufman Adaptive Moving Average (KAMA) - 10 period
    def calculate_kama(close, period=10):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[period:] = change[period-1:] / volatility[period-1:]
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        kama = np.zeros_like(close)
        kama[:period] = close[:period]
        for i in range(period, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Relative Strength Index (RSI) - 14 period
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Choppiness Index (CHOP) - 14 period
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros_like(close)
        for i in range(period, len(close)):
            atr[i] = np.sum(tr[i-period+1:i+1])
        # Maximum and minimum close over period
        max_hc = np.zeros_like(close)
        min_lc = np.zeros_like(close)
        for i in range(period-1, len(close)):
            max_hc[i] = np.max(high[i-period+1:i+1])
            min_lc[i] = np.min(low[i-period+1:i+1])
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if max_hc[i] != min_lc[i]:
                chop[i] = 100 * np.log10(atr[i] / (max_hc[i] - min_lc[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral when no range
        return chop
    
    # Calculate indicators on daily data
    kama_1d = calculate_kama(close_1d, 10)
    rsi_1d = calculate_rsi(close_1d, 14)
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, close_1d, 14)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend direction
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align to daily timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(ema21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: price above/below EMA21
        weekly_uptrend = close_1d[i] > ema21_1w_aligned[i]
        weekly_downtrend = close_1d[i] < ema21_1w_aligned[i]
        
        # Daily conditions
        price_above_kama = close[i] > kama_1d_aligned[i]
        price_below_kama = close[i] < kama_1d_aligned[i]
        rsi_overbought = rsi_1d_aligned[i] > 70
        rsi_oversold = rsi_1d_aligned[i] < 30
        market_trending = chop_1d_aligned[i] < 38.2  # trending market
        
        # Entry logic: follow weekly trend, use RSI extremes in trending markets
        if weekly_uptrend and rsi_oversold and market_trending and position != 1:
            # Buy dips in uptrend
            position = 1
            signals[i] = 0.25
        elif weekly_downtrend and rsi_overbought and market_trending and position != -1:
            # Sell rallies in downtrend
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (not weekly_uptrend or rsi_overbought or chop_1d_aligned[i] > 61.8):
            # Exit long if trend turns, RSI overbought, or market becomes choppy
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not weekly_downtrend or rsi_oversold or chop_1d_aligned[i] > 61.8):
            # Exit short if trend turns, RSI oversold, or market becomes choppy
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0