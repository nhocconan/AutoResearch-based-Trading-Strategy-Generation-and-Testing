#!/usr/bin/env python3
"""
Hypothesis: Daily RSI(14) with 200-day SMA trend filter and weekly volatility filter.
Enters long when RSI crosses above 30 from below with price above 200-day SMA and weekly ATR contraction.
Enters short when RSI crosses below 70 from above with price below 200-day SMA and weekly ATR contraction.
Uses weekly ATR contraction (current ATR < 0.8 * ATR 4 weeks ago) to identify low volatility periods for mean reversion.
Targets 10-25 trades/year per symbol (40-100 total over 4 years) to minimize fee drag.
Works in both bull and bear markets by combining mean reversion with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # First average
    if len(close) >= period:
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        # Subsequent values
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr_val = np.zeros_like(close)
    if len(close) >= period:
        atr_val[period-1] = np.mean(tr[:period])
        for i in range(period, len(close)):
            atr_val[i] = (atr_val[i-1] * (period-1) + tr[i]) / period
    return atr_val

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for RSI and SMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for ATR volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    rsi_val = rsi(close, 14)
    
    # Calculate daily SMA(200) for trend filter
    sma_200 = np.full_like(close, np.nan)
    if len(close) >= 200:
        sma_200[199] = np.mean(close[:200])
        for i in range(200, len(close)):
            sma_200[i] = np.mean(close[i-199:i+1])
    
    # Calculate weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = atr(high_1w, low_1w, close_1w, 14)
    
    # Align indicators to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_val)
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI, SMA, and ATR
    start_idx = max(14, 200, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(sma_200_aligned[i]) or 
            np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        rsi_now = rsi_aligned[i]
        rsi_prev = rsi_aligned[i-1]
        sma_200_now = sma_200_aligned[i]
        atr_now = atr_1w_aligned[i]
        price_now = close[i]
        
        # Weekly ATR contraction: current ATR < 0.8 * ATR 4 weeks ago
        # Need to look back 4 weeks in weekly data
        atr_contraction = False
        if i >= 28:  # Need at least 4 weeks of data (approximate)
            # Get ATR from approximately 4 weeks ago (28 days)
            idx_4w_ago = i - 28
            if idx_4w_ago >= 0 and not np.isnan(atr_1w_aligned[idx_4w_ago]):
                atr_4w_ago = atr_1w_aligned[idx_4w_ago]
                atr_contraction = atr_now < 0.8 * atr_4w_ago
        
        # RSI crossing conditions
        rsi_cross_above_30 = rsi_prev <= 30 and rsi_now > 30
        rsi_cross_below_70 = rsi_prev >= 70 and rsi_now < 70
        
        # Trend filter
        above_sma200 = price_now > sma_200_now
        below_sma200 = price_now < sma_200_now
        
        # Entry conditions
        if position == 0:
            # Long: RSI crosses above 30 + price above SMA200 + ATR contraction
            if rsi_cross_above_30 and above_sma200 and atr_contraction:
                signals[i] = size
                position = 1
            # Short: RSI crosses below 70 + price below SMA200 + ATR contraction
            elif rsi_cross_below_70 and below_sma200 and atr_contraction:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses below 50 or price closes below SMA200
            if rsi_now < 50 or price_now < sma_200_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI crosses above 50 or price closes above SMA200
            if rsi_now > 50 or price_now > sma_200_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "daily_RSI14_SMA200_ATRcontraction"
timeframe = "1d"
leverage = 1.0