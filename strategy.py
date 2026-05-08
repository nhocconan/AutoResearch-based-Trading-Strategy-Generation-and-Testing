#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA direction + RSI + Chop regime filter
# KAMA adapts to market noise, providing smooth trend direction.
# RSI(14) filters for momentum strength (avoids choppy reversals).
# Chop regime filter (Chop > 61.8 = range, Chop < 38.2 = trend) ensures we only
# trade in trending markets, avoiding whipsaws in sideways action.
# This combination works in bull markets (trend + momentum) and bear markets
# (avoids false breaks in chop, only takes strong directional moves).
# Targets ~20-30 trades per year (~80-120 total over 4 years) to minimize fee drag.

name = "4h_KAMA_RSI_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend direction
    def calculate_kama(close, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 2, 30)
    kama_dir = kama > np.roll(kama, 1)  # 1 if rising, 0 if falling
    
    # RSI(14) - momentum filter
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Wilder smoothing
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Chop regime filter (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14)
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Chop calculation
    def calculate_chop(high, low, close, atr, period=14):
        # Sum of true range over period
        tr_sum = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                tr_sum[i] = np.sum(tr[:i+1]) if i > 0 else tr[0]
            else:
                tr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                highest_high[i] = np.max(high[:i+1])
                lowest_low[i] = np.min(low[:i+1])
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop formula
        chop = np.where(
            (highest_high - lowest_low) != 0,
            100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period),
            50
        )
        return chop
    
    chop = calculate_chop(high_1d, low_1d, close_1d, atr, 14)
    chop_trend = chop < 38.2  # Trending regime
    chop_range = chop > 61.8   # Ranging regime
    
    # Align indicators to 4h timeframe
    kama_dir_4h = align_htf_to_ltf(prices, df_1d, kama_dir.astype(float))
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    chop_trend_4h = align_htf_to_ltf(prices, df_1d, chop_trend.astype(float))
    chop_range_4h = align_htf_to_ltf(prices, df_1d, chop_range.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_dir_4h[i]) or np.isnan(rsi_4h[i]) or 
            np.isnan(chop_trend_4h[i]) or np.isnan(chop_range_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising, RSI > 50, trending regime
            if kama_dir_4h[i] and rsi_4h[i] > 50 and chop_trend_4h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI < 50, trending regime
            elif not kama_dir_4h[i] and rsi_4h[i] < 50 and chop_trend_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA falling OR RSI < 40 OR ranging regime
            if (not kama_dir_4h[i]) or (rsi_4h[i] < 40) or chop_range_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA rising OR RSI > 60 OR ranging regime
            if kama_dir_4h[i] or (rsi_4h[i] > 60) or chop_range_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals