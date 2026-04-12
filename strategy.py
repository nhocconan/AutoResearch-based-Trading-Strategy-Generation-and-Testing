#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA with RSI and Chop Filter
# Uses Kaufman Adaptive Moving Average (KAMA) on daily data to determine trend direction.
# Long when price > KAMA and RSI < 50 (avoid overbought), short when price < KAMA and RSI > 50 (avoid oversold).
# Only trade when Choppiness Index > 61.8 (ranging market) to avoid whipsaws in strong trends.
# Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag.
# Works in ranging markets via mean reversion to KAMA and avoids trending markets where KAMA lags.
# Focus on BTC/ETH as primary targets.

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate daily KAMA (10, 2, 30)
    def kama(close, fast=2, slow=30, lookback=10):
        # Efficiency Ratio
        change = np.abs(np.diff(close, lookback))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[lookback:] = change[lookback:] / volatility[lookback:]
        er[:lookback] = 0
        
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
        
        # KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, 2, 30, 10)
    
    # Calculate daily RSI (14)
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
        
        # Wilder smoothing
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Calculate weekly Choppiness Index (14)
    def chop(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of True Range over period
        atr_sum = np.zeros_like(close)
        for i in range(period, len(tr)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(period, len(high)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop_vals = np.zeros_like(close)
        for i in range(period, len(close)):
            if atr_sum[i] > 0 and max_high[i] != min_low[i]:
                chop_vals[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop_vals[i] = 50  # neutral
        return chop_vals
    
    chop_vals = chop(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    chop_align = align_htf_to_ltf(prices, df_1w, chop_vals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_align[i]):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (Chop > 61.8)
        if chop_align[i] <= 61.8:
            # Hold current position if not ranging
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price above KAMA and RSI < 50 (not overbought)
        if close[i] > kama_vals[i] and rsi_vals[i] < 50 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price below KAMA and RSI > 50 (not oversold)
        elif close[i] < kama_vals[i] and rsi_vals[i] > 50 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price crosses KAMA (mean reversion)
        elif position == 1 and close[i] <= kama_vals[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= kama_vals[i]:
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