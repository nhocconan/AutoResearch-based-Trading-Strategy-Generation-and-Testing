#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v1
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
Combine with RSI for momentum confirmation and Choppiness Index to avoid whipsaws in ranging markets.
Long when KAMA turns up, RSI > 50, and market is trending (CHOP < 38.2).
Short when KAMA turns down, RSI < 50, and market is trending (CHOP < 38.2).
Target: 15-25 trades/year by requiring trend alignment and momentum confirmation.
Works in bull markets via trend following and in bear markets via short signals during downtrends.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (adaptive moving average)
    def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[er_period:] = change[er_period-1:] / volatility[er_period-1:]
        er[er == 0] = 0
        
        # Smoothing Constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1))**2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    
    # Calculate RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        if len(close) > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.zeros_like(close)
        rs[period:] = avg_gain[period:] / np.where(avg_loss[period:] == 0, 1e-10, avg_loss[period:])
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Calculate Choppiness Index
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR
        atr = np.zeros_like(tr)
        for i in range(1, len(tr)):
            if i <= period:
                atr[i] = np.mean(tr[1:i+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Sum of ATR over period
        atr_sum = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                atr_sum[i] = np.nan
            else:
                atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros_like(high)
        min_low = np.zeros_like(low)
        for i in range(len(close)):
            if i < period:
                max_high[i] = np.nan
                min_low[i] = np.nan
            else:
                max_high[i] = np.max(high[i-period+1:i+1])
                min_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if i < period or np.isnan(atr_sum[i]) or (max_high[i] - min_low[i]) == 0:
                chop[i] = np.nan
            else:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly EMA to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 14) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters: price relative to weekly EMA and chop < 38.2 (trending market)
        price_above_ema = close[i] > ema_1w_aligned[i]
        price_below_ema = close[i] < ema_1w_aligned[i]
        trending_market = chop[i] < 38.2
        
        if position == 0:
            # Long: KAMA turning up, RSI > 50, trending market, price above weekly EMA
            if (i > 0 and kama[i] > kama[i-1] and 
                rsi[i] > 50 and trending_market and price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, RSI < 50, trending market, price below weekly EMA
            elif (i > 0 and kama[i] < kama[i-1] and 
                  rsi[i] < 50 and trending_market and price_below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turning down OR chop > 61.8 (ranging market)
            if (i > 0 and kama[i] < kama[i-1]) or chop[i] > 61.8:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turning up OR chop > 61.8 (ranging market)
            if (i > 0 and kama[i] > kama[i-1]) or chop[i] > 61.8:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0