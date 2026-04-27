#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA (Kaufman Adaptive Moving Average) with RSI and chop filter.
# Uses KAMA to capture trend direction (adaptive smoothing reduces whipsaws).
# RSI(14) for overbought/oversold conditions with mean reversion in ranging markets.
# Choppiness Index (CHOP) to filter regimes: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (follow trend).
# Long when: KAMA rising, RSI < 40 (oversold), and CHOP > 50 (ranging bias).
# Short when: KAMA falling, RSI > 60 (overbought), and CHOP > 50 (ranging bias).
# Exit when opposite signal or CHOP < 40 (strong trend - avoid mean reversion in trends).
# Designed for ~10-20 trades/year with low frequency to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly KAMA for trend filter
    def kama(series, period=10, fast=2, slow=30):
        """Kaufman Adaptive Moving Average"""
        if len(series) < period:
            return np.full_like(series, np.nan)
        change = np.abs(np.diff(series, period))
        volatility = np.abs(np.diff(series)).rolling(window=period).sum()
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_vals = np.full_like(series, np.nan)
        kama_vals[period-1] = series[period-1]
        for i in range(period, len(series)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (series[i] - kama_vals[i-1])
        return kama_vals
    
    kama_1w = kama(close_1w)
    
    # Calculate daily indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA on daily data
    kama_1d = kama(close_1d)
    
    # Calculate RSI(14)
    def rsi(series, period=14):
        delta = np.diff(series)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(series, np.nan)
        avg_loss = np.full_like(series, np.nan)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(series)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_1d = rsi(close_1d)
    
    # Calculate Choppiness Index
    def choppiness_index(high, low, close, period=14):
        """Choppiness Index: higher = ranging, lower = trending"""
        atr = np.zeros_like(close)
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        
        # smoothed ATR
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(period-1, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        # chop calculation
        chop = np.full_like(close, 50.0)
        for i in range(period-1, len(close)):
            if atr_sum[i] > 0 and hh[i] != ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
        return chop
    
    chop_1d = choppiness_index(high_1d, low_1d, close_1d)
    
    # Align weekly indicators to daily
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Align daily indicators to 1d timeframe (no alignment needed as we're using 1d data directly)
    # But we need to align to the 1d index in prices
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Weekly trend filter: price above/below weekly KAMA
        weekly_uptrend = price > kama_1w_aligned[i]
        weekly_downtrend = price < kama_1w_aligned[i]
        
        # Daily KAMA slope (direction)
        kama_rising = kama_1d_aligned[i] > kama_1d_aligned[i-1]
        kama_falling = kama_1d_aligned[i] < kama_1d_aligned[i-1]
        
        # RSI conditions
        rsi_oversold = rsi_1d_aligned[i] < 40
        rsi_overbought = rsi_1d_aligned[i] > 60
        
        # Chop filter: >50 = ranging bias (good for mean reversion)
        chop_ranging = chop_1d_aligned[i] > 50
        chop_trending = chop_1d_aligned[i] < 40  # strong trend - avoid mean reversion
        
        if position == 0:
            # Long: weekly uptrend, KAMA rising, RSI oversold, ranging market
            if weekly_uptrend and kama_rising and rsi_oversold and chop_ranging:
                signals[i] = size
                position = 1
            # Short: weekly downtrend, KAMA falling, RSI overbought, ranging market
            elif weekly_downtrend and kama_falling and rsi_overbought and chop_ranging:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: opposite signal or strong trend (avoid mean reversion in trends)
            if (not weekly_uptrend or not kama_rising or not rsi_oversold) or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: opposite signal or strong trend
            if (not weekly_downtrend or not kama_falling or not rsi_overbought) or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_MeanReversion"
timeframe = "1d"
leverage = 1.0