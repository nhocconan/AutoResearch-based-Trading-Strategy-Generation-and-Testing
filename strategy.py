#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily KAMA with RSI and Choppiness Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction.
# In trending markets (ADX > 25), KAMA signals capture momentum. In ranging markets
# (Choppiness > 61.8), RSI mean reversion at extremes (30/70) takes over.
# Daily trend filter avoids whipsaws. Target: 20-35 trades/year (80-140 total).

name = "4h_daily_kama_rsi_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and regime filters
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Daily KAMA (trend direction)
    daily_close = df_daily['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(daily_close, n=10))  # 10-period change
    abs_change = np.abs(np.diff(daily_close))     # daily absolute change
    er = np.zeros_like(daily_close)
    er[10:] = change[9:] / (np.nansum(abs_change.reshape(-1, 10), axis=1) + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(daily_close, np.nan)
    kama[29] = np.mean(daily_close[0:30])  # seed
    for i in range(30, len(daily_close)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (daily_close[i] - kama[i-1])
    
    # Daily ADX for trend strength
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    # True Range
    tr1 = daily_high[1:] - daily_low[1:]
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((daily_high[1:] - daily_high[:-1]) > (daily_low[:-1] - daily_low[1:]),
                       np.maximum(daily_high[1:] - daily_high[:-1], 0), 0)
    dm_minus = np.where((daily_low[:-1] - daily_low[1:]) > (daily_high[1:] - daily_high[:-1]),
                        np.maximum(daily_low[:-1] - daily_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[1:period]) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    atr_d = wilders_smoothing(tr, 14)
    dm_plus_s = wilders_smoothing(dm_plus, 14)
    dm_minus_s = wilders_smoothing(dm_minus, 14)
    di_plus = np.where(atr_d > 0, 100 * dm_plus_s / atr_d, 0)
    di_minus = np.where(atr_d > 0, 100 * dm_minus_s / atr_d, 0)
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_daily = wilders_smoothing(dx, 14)
    
    # Daily Choppiness Index (regime filter)
    # Chop = log10(sum(TR, n) / (max(high, n) - min(low, n))) / log10(n) * 100
    n_chop = 14
    tr_sum = np.convolve(tr, np.ones(n_chop), 'same')
    tr_sum[:n_chop-1] = np.nan
    max_high = np.zeros_like(daily_high)
    min_low = np.zeros_like(daily_low)
    for i in range(n_chop, len(daily_high)):
        max_high[i] = np.max(daily_high[i-n_chop+1:i+1])
        min_low[i] = np.min(daily_low[i-n_chop+1:i+1])
    max_high[:n_chop-1] = np.nan
    min_low[:n_chop-1] = np.nan
    range_n = max_high - min_low
    chop = np.zeros_like(daily_close)
    chop = np.where(range_n > 0, np.log10(tr_sum / range_n) / np.log10(n_chop) * 100, 50)
    
    # 4h RSI for entry timing
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily indicators to 4h
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx_daily)
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(kama_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend weakens or RSI overbought
            if adx_aligned[i] < 20 or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: trend weakens or RSI oversold
            if adx_aligned[i] < 20 or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Determine regime: trending (Chop <= 61.8) or ranging (Chop > 61.8)
            if chop_aligned[i] <= 61.8:  # Trending regime
                if adx_aligned[i] >= 25:  # Strong trend
                    # Long: price above KAMA
                    if close[i] > kama_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price below KAMA
                    elif close[i] < kama_aligned[i]:
                        position = -1
                        signals[i] = -0.25
            else:  # Ranging regime
                # Mean reversion at RSI extremes
                if rsi[i] < 30:  # Oversold -> long
                    position = 1
                    signals[i] = 0.25
                elif rsi[i] > 70:  # Overbought -> short
                    position = -1
                    signals[i] = -0.25
    
    return signals