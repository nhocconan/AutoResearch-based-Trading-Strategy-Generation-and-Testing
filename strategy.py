#!/usr/bin/env python3
"""
1D_KAMA_Trend_RSI_Chop_v2
Hypothesis: Kaufman Adaptive Moving Average (KAMA) for trend direction on 1d,
combined with RSI(14) mean-reversion signals and Choppiness Index regime filter.
Only takes long signals in uptrend (price > KAMA) and short in downtrend (price < KAMA).
Uses Choppiness Index to avoid whipsaws in ranging markets (CHOP > 61.8 = range, avoid trend trades).
Designed for low trade frequency (<25/year) with discrete sizing (0.25) to minimize fee drag.
Works in both bull and bear markets by following trend direction and avoiding counter-trend trades.
"""

name = "1D_KAMA_Trend_RSI_Chop_v2"
timeframe = "1d"
leverage = 1.0

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
    
    # Get 1w data for trend filter (KAMA on weekly)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate KAMA(30) on weekly close
    def kama(close_series, period=30, fast=2, slow=30):
        close_series = pd.Series(close_series)
        change = abs(close_series.diff(period))
        volatility = close_series.diff().abs().rolling(window=period, min_periods=period).sum()
        er = change / volatility.replace(0, np.nan)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
        sc = sc.fillna(0)
        kama = [np.nan] * len(close_series)
        kama[period-1] = close_series.iloc[period-1]
        for i in range(period, len(close_series)):
            kama[i] = kama[i-1] + sc[i] * (close_series.iloc[i] - kama[i-1])
        return np.array(kama)
    
    kama_1w = kama(df_1w['close'].values, period=30, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate RSI(14) on daily close
    def rsi(close_series, period=14):
        close_series = pd.Series(close_series)
        delta = close_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    
    rsi_14 = rsi(close, period=14)
    
    # Calculate Choppiness Index(14) on daily data
    def chop(high_series, low_series, close_series, period=14):
        high_series = pd.Series(high_series)
        low_series = pd.Series(low_series)
        close_series = pd.Series(close_series)
        atr = np.maximum(
            high_series - low_series,
            np.maximum(
                abs(high_series - close_series.shift(1)),
                abs(low_series - close_series.shift(1))
            )
        )
        atr_sum = atr.rolling(window=period, min_periods=period).sum()
        highest_high = high_series.rolling(window=period, min_periods=period).max()
        lowest_low = low_series.rolling(window=period, min_periods=period).min()
        range_hl = highest_high - lowest_low
        chop = 100 * np.log10(atr_sum / range_hl.replace(0, np.nan)) / np.log10(period)
        return chop.fillna(50).values
    
    chop_14 = chop(high, low, close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Warmup for KAMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from weekly KAMA
        price_above_kama = close[i] > kama_1w_aligned[i]
        price_below_kama = close[i] < kama_1w_aligned[i]
        
        # Choppiness filter: only trade when NOT in strong range (CHOP <= 61.8)
        not_ranging = chop_14[i] <= 61.8
        
        if position == 0:
            # Long entry: price above weekly KAMA + RSI < 40 (oversold) + not ranging
            if (price_above_kama and 
                rsi_14[i] < 40 and 
                not_ranging):
                signals[i] = 0.25
                position = 1
            # Short entry: price below weekly KAMA + RSI > 60 (overbought) + not ranging
            elif (price_below_kama and 
                  rsi_14[i] > 60 and 
                  not_ranging):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below weekly KAMA OR RSI > 60 (overbought)
            if (close[i] < kama_1w_aligned[i] or rsi_14[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above weekly KAMA OR RSI < 40 (oversold)
            if (close[i] > kama_1w_aligned[i] or rsi_14[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals