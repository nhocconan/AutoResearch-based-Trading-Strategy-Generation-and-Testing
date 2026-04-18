#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) as trend filter on 1d timeframe, combined with RSI momentum and Choppiness Index regime filter to avoid whipsaws. KAMA adapts to market noise, reducing false signals in ranging markets while capturing trends. RSI provides entry timing, and Choppiness Index filters out low-quality signals in choppy regimes. This combination should work in both bull and bear markets by only taking trades when trend is clear (KAMA slope) and market is not choppy. Targets 10-20 trades/year by requiring KAMA alignment, RSI extreme, and low chop (< 38.2). Uses 1w timeframe for higher context trend filter to avoid counter-trend trades.
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
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA (adaptive moving average)
    # Parameters: ER period=10, Fast EMA=2, Slow EMA=30
    def kama(close, er_period=10, fast=2, slow=30):
        n = len(close)
        kama_out = np.full(n, np.nan)
        if n < er_period + 1:
            return kama_out
        
        # Efficiency Ratio
        change = np.abs(np.diff(close, er_period))  # |close[t] - close[t-er_period]|
        volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over er_period
        
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Smoothing constants
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        
        # Initialize KAMA
        kama_out[er_period] = close[er_period]
        
        for i in range(er_period + 1, n):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        
        return kama_out
    
    kama_1d = kama(close_1d, 10, 2, 30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(14) on 1d
    def rsi(close, period=14):
        n = len(close)
        rsi_out = np.full(n, np.nan)
        if n < period + 1:
            return rsi_out
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        # Initial average
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    rsi_1d = rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Simple trend: price > EMA34 on weekly
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = close_1w[i] * (2/(34+1)) + ema_34_1w[i-1] * (1 - 2/(34+1))
    
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Choppiness Index(14) on 1d
    def choppy(high, low, close, period=14):
        n = len(close)
        chop_out = np.full(n, np.nan)
        if n < period + 1:
            return chop_out
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Sum of TR over period
        tr_sum = np.full(n, np.nan)
        for i in range(period, n):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.full(n, np.nan)
        min_low = np.full(n, np.nan)
        for i in range(period-1, n):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop = 100 * log10(sum(tr) / (max_high - min_low)) / log10(period)
        range_hl = max_high - min_low
        chop_out = np.where(
            (range_hl > 0) & ~np.isnan(tr_sum),
            100 * np.log10(tr_sum) / np.log10(period) / np.log10(range_hl),
            50  # default to middle when range is zero
        )
        return chop_out
    
    chop_1d = choppy(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime: < 38.2 = trending, > 61.8 = ranging
        is_trending = chop_1d_aligned[i] < 38.2
        
        if position == 0:
            # Long entry: price > KAMA (uptrend), RSI > 50 (momentum), weekly EMA uptrend, and trending market
            if (close[i] > kama_1d_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                close[i] > ema_34_1w_aligned[i] and
                is_trending):
                signals[i] = 0.25
                position = 1
            # Short entry: price < KAMA (downtrend), RSI < 50 (momentum), weekly EMA downtrend, and trending market
            elif (close[i] < kama_1d_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  close[i] < ema_34_1w_aligned[i] and
                  is_trending):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price < KAMA (trend change) or RSI < 40 (loss of momentum) or chop > 61.8 (ranging)
            if (close[i] < kama_1d_aligned[i] or 
                rsi_1d_aligned[i] < 40 or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA (trend change) or RSI > 60 (loss of momentum) or chop > 61.8 (ranging)
            if (close[i] > kama_1d_aligned[i] or 
                rsi_1d_aligned[i] > 60 or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0