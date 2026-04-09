#!/usr/bin/env python3
# 1d_weekly_kama_rsi_chop_v1
# Hypothesis: 1d strategy combining weekly trend filter (price above/below weekly KAMA) with daily KAMA/RSI entries and choppiness regime filter.
# Long: Weekly bullish trend (close > weekly KAMA) + daily KAMA rising + RSI < 30 + chop regime (CHOP > 61.8 = ranging)
# Short: Weekly bearish trend (close < weekly KAMA) + daily KAMA falling + RSI > 70 + chop regime (CHOP > 61.8 = ranging)
# Exit: Opposite signal or weekly trend flip.
# Uses 1d primary timeframe with 1w HTF for trend and chop regime.
# Designed for low trade frequency (~10-25/year) to minimize fee drag while capturing mean reversion in ranging markets and pullbacks in trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA efficiency ratio calculation
    def calculate_kama(arr, period=10, fast=2, slow=30):
        change = np.abs(np.diff(arr, period))
        volatility = np.sum(np.abs(np.diff(arr)), axis=0) if arr.ndim > 1 else np.sum(np.abs(np.diff(arr)))
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(arr, np.nan, dtype=float)
        kama[period] = arr[period]
        for i in range(period+1, len(arr)):
            kama[i] = kama[i-1] + sc[i] * (arr[i] - kama[i-1])
        return kama
    
    # Choppiness Index calculation
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        chop = np.where((highest_high - lowest_low) != 0,
                        100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period),
                        50)
        return chop
    
    # Get 1w data for trend filter and chop regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly KAMA for trend filter
    kama_1w = calculate_kama(close_1w, period=10, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Weekly choppy regime filter
    chop_1w = calculate_chop(high_1w, low_1w, close_1w, period=14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w, additional_delay_bars=0)
    
    # Get 1d data for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily KAMA for entry
    kama_1d = calculate_kama(close_1d, period=10, fast=2, slow=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Daily RSI for overbought/oversold
    delta = pd.Series(close_1d).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(chop_1w_aligned[i]) or
            np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Choppiness regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
        # We use chop regime as filter: only trade when ranging (CHOP > 61.8) for mean reversion
        in_chop_regime = chop_1w_aligned[i] > 61.8
        
        # Daily KAMA direction
        kama_rising = kama_1d_aligned[i] > kama_1d_aligned[i-1] if i > 0 else False
        kama_falling = kama_1d_aligned[i] < kama_1d_aligned[i-1] if i > 0 else False
        
        if position == 1:  # Long position
            # Exit: Opposite signal or weekly trend turns bearish
            if (close[i] < kama_1d_aligned[i] or  # Price below daily KAMA
                close[i] < kama_1w_aligned[i]):   # Weekly trend turns bearish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Opposite signal or weekly trend turns bullish
            if (close[i] > kama_1d_aligned[i] or  # Price above daily KAMA
                close[i] > kama_1w_aligned[i]):   # Weekly trend turns bullish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Weekly bullish trend + daily KAMA rising + RSI oversold + chop regime
            if (close[i] > kama_1w_aligned[i] and  # Weekly bullish trend
                kama_rising and                    # Daily KAMA rising
                rsi_1d_aligned[i] < 30 and         # RSI oversold
                in_chop_regime):                   # Choppy/ranging regime
                position = 1
                signals[i] = 0.25
            # Short entry: Weekly bearish trend + daily KAMA falling + RSI overbought + chop regime
            elif (close[i] < kama_1w_aligned[i] and  # Weekly bearish trend
                  kama_falling and                   # Daily KAMA falling
                  rsi_1d_aligned[i] > 70 and         # RSI overbought
                  in_chop_regime):                   # Choppy/ranging regime
                position = -1
                signals[i] = -0.25
    
    return signals