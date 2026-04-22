#!/usr/bin/env python3
"""
Hypothesis: Daily KAMA + RSI + Chop Filter
Long when KAMA is rising and RSI > 50 in low chop (trending market), short when KAMA is falling and RSI < 50 in low chop.
Uses 1-week ADX to filter ranging markets (ADX < 20). Designed for low trade frequency by requiring trend alignment.
Works in both bull and bear markets by following the weekly trend via ADX and avoiding choppy markets.
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
    
    # KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_len=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize KAMA
        kama = np.full_like(close, np.nan, dtype=float)
        kama[er_len] = close[er_len]
        for i in range(er_len + 1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_val = kama(close)
    
    # RSI (14)
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close, np.nan, dtype=float)
        avg_loss = np.full_like(close, np.nan, dtype=float)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_val = rsi(close)
    
    # Chopiness Index (14)
    def chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.convolve(tr, np.ones(period)/period, mode='same')
        atr[:period-1] = np.nan
        atr[-(period-1):] = np.nan
        # True range sum over period
        tr_sum = np.convolve(tr, np.ones(period)/period, mode='same')
        tr_sum[:period-1] = np.nan
        tr_sum[-(period-1):] = np.nan
        # Chop formula: 100 * log10(tr_sum / (atr * period)) / log10(period)
        chop_val = 100 * np.log10(tr_sum / (atr * period)) / np.log10(period)
        return chop_val
    
    chop_val = chop(high, low, close)
    
    # Load 1-week data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # ADX (14) on weekly
    def adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        # Smooth TR, DM+
        tr_period = np.convolve(tr, np.ones(period)/period, mode='same')
        dm_plus_period = np.convolve(dm_plus, np.ones(period)/period, mode='same')
        dm_minus_period = np.convolve(dm_minus, np.ones(period)/period, mode='same')
        tr_period[:period-1] = np.nan
        dm_plus_period[:period-1] = np.nan
        dm_minus_period[:period-1] = np.nan
        # DI+ and DI-
        di_plus = 100 * dm_plus_period / tr_period
        di_minus = 100 * dm_minus_period / tr_period
        # DX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        # ADX
        adx = np.convolve(dx, np.ones(period)/period, mode='same')
        adx[:2*period-2] = np.nan
        return adx
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_1w = adx(high_1w, low_1w, close_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or np.isnan(chop_val[i]) or 
            np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade when ADX >= 20 (trending market)
        trending = adx_1w_aligned[i] >= 20
        
        if position == 0:
            # Long: KAMA rising and RSI > 50 in trending market
            if kama_val[i] > kama_val[i-1] and rsi_val[i] > 50 and trending:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling and RSI < 50 in trending market
            elif kama_val[i] < kama_val[i-1] and rsi_val[i] < 50 and trending:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: KAMA reverses or ADX drops below 20 (ranging market)
            exit_signal = False
            
            if position == 1:
                # Exit long: KAMA turns down or market ranges
                if kama_val[i] <= kama_val[i-1] or not trending:
                    exit_signal = True
            else:  # position == -1
                # Exit short: KAMA turns up or market ranges
                if kama_val[i] >= kama_val[i-1] or not trending:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0