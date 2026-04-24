#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA + RSI(2) + Choppiness Index regime filter.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for KAMA trend direction.
- KAMA: Kaufman Adaptive Moving Average with ER=10, fast=2, slow=30.
- RSI(2): Ultra-short RSI for mean-reversion entries.
- Choppiness Index: CHOP(14) > 61.8 = range (mean revert), CHOP < 38.2 = trending (avoid).
- Entry: Long when KAMA up, RSI(2)<10, and CHOP>61.8 (oversold in range).
         Short when KAMA down, RSI(2)>90, and CHOP>61.8 (overbought in range).
- Exit: RSI(2)>50 for long exit, RSI(2)<50 for short exit.
- Signal size: 0.25 discrete to minimize fee drag.
- Works in ranging markets (2025+) by fading extremes in chop, avoids trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w KAMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need sufficient data for KAMA
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # KAMA(ER=10, fast=2, slow=30)
    def kama(close, er_period=10, fast=2, slow=30):
        n = len(close)
        kama_vals = np.full(n, np.nan)
        if n < er_period + 1:
            return kama_vals
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_vals[er_period] = close[er_period]
        for i in range(er_period + 1, n):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_1w = kama(close_1w)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate RSI(2) on 1d
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=2, min_periods=2).mean().values
    avg_loss = pd.Series(loss).rolling(window=2, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi2 = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14) on 1d
    def chop(high, low, close, window=14):
        n = len(close)
        chop_vals = np.full(n, np.nan)
        atr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
        atr[0] = high[0] - low[0]
        sum_atr = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        range_ = highest_high - lowest_low
        chop_vals = 100 * np.log10(sum_atr / range_) / np.log10(window)
        return chop_vals
    
    chop_vals = chop(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 2)  # Need 30 for KAMA, 2 for RSI2
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi2[i]) or np.isnan(chop_vals[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_rsi2 = rsi2[i]
        curr_chop = chop_vals[i]
        curr_kama = kama_1w_aligned[i]
        
        # KAMA trend: compare current KAMA to previous KAMA
        kama_up = curr_kama > kama_1w_aligned[i-1] if i > 0 else False
        kama_down = curr_kama < kama_1w_aligned[i-1] if i > 0 else False
        
        # Choppiness regime: CHOP > 61.8 = range (good for mean reversion)
        in_range = curr_chop > 61.8
        
        # Exit conditions: RSI(2) mean reversion
        if position != 0:
            # Exit long: RSI(2) > 50
            if position == 1:
                if curr_rsi2 > 50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: RSI(2) < 50
            elif position == -1:
                if curr_rsi2 < 50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: KAMA trend + RSI(2) extreme + range regime
        if position == 0:
            # Long: KAMA up, RSI(2) < 10 (oversold), and in range
            long_condition = kama_up and curr_rsi2 < 10 and in_range
            
            # Short: KAMA down, RSI(2) > 90 (overbought), and in range
            short_condition = kama_down and curr_rsi2 > 90 and in_range
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI2_ChopRegime_v1"
timeframe = "1d"
leverage = 1.0