#!/usr/bin/env python3
"""
12h KAMA Trend + RSI Mean Reversion + Daily Chop Filter
Hybrid strategy combining trend following (KAMA) with mean reversion (RSI) 
filtered by daily market regime (Choppiness Index). Designed to work in both 
bull and bear markets by adapting to trending vs ranging conditions.
Uses 12h timeframe with daily Choppiness Index filter to avoid false signals.
Target: 15-30 trades/year with clear entry/exit rules to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA on 12h data for trend direction
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Kaufman Adaptive Moving Average (KAMA) - 14 period
    def calculate_kama(close, period=14, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[period:] = change[period-1:] / volatility[period-1:]
        er[np.isnan(er)] = 0
        er[volatility == 0] = 0
        
        # Smoothing Constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        sc[np.isnan(sc)] = 0
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_12h = calculate_kama(close_12h, period=14, fast=2, slow=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate RSI (14) on price
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
    
    rsi = calculate_rsi(close, period=14)
    
    # Calculate Choppiness Index on daily data
    def calculate_choppiness(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            if i < period:
                atr[i] = np.nan
            else:
                atr[i] = np.nanmean(tr[i-period+1:i+1])
        
        # Sum of true ranges over period
        sum_tr = np.zeros_like(close)
        for i in range(len(close)):
            if i < period-1:
                sum_tr[i] = np.nan
            else:
                sum_tr[i] = np.nansum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(len(close)):
            if i < period-1:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.nanmax(high[i-period+1:i+1])
                lowest_low[i] = np.nanmin(low[i-period+1:i+1])
        
        # Choppiness Index
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if np.isnan(sum_tr[i]) or atr[i] == 0 or highest_high[i] == lowest_low[i]:
                chop[i] = np.nan
            else:
                chop[i] = 100 * np.log10(sum_tr[i] / (atr[i] * period)) / np.log10(period)
        return chop
    
    chop = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama_12h_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama = kama_12h_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop_aligned[i]
        
        # Market regime: chop > 61.8 = ranging, chop < 38.2 = trending
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        if position == 0:
            # In ranging markets: mean reversion at RSI extremes
            if is_ranging:
                if rsi_val < 30:  # oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi_val > 70:  # overbought
                    signals[i] = -0.25
                    position = -1
            # In trending markets: follow KAMA direction
            elif is_trending:
                if price > kama and rsi_val > 50:  # bullish
                    signals[i] = 0.25
                    position = 1
                elif price < kama and rsi_val < 50:  # bearish
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit conditions
            if is_ranging and rsi_val > 50:  # mean reversion exit in ranging
                signals[i] = 0.0
                position = 0
            elif is_trending and price < kama:  # trend reversal exit
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit conditions
            if is_ranging and rsi_val < 50:  # mean reversion exit in ranging
                signals[i] = 0.0
                position = 0
            elif is_trending and price > kama:  # trend reversal exit
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0