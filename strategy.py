#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d/1w pivot-based breakout/mean-reversion hybrid.
Long when price breaks above 1d R1 with 1w EMA50 > EMA200 and volume > 1.3x 20-period 6h volume average.
Short when price breaks below 1d S1 with 1w EMA50 < EMA200 and volume > 1.3x 20-period 6h volume average.
Mean-reversion fade when price touches 1d R3/S3 with RSI(14) > 70/<30 and volume < 0.8x average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Combines structural pivot levels, multi-timeframe trend filter, volume confirmation, and RSI extremes.
Designed to work in bull markets (breakout continuation) and bear markets (strong trend continuation or mean reversion at extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Get 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1w EMA50 and EMA200 for trend
    def ema(values, span):
        return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema_50_1w = ema(close_1w, 50)
    ema_200_1w = ema(close_1w, 200)
    
    # Calculate 1d pivot points (standard formula)
    def calculate_pivots(high_vals, low_vals, close_vals):
        pivot = (high_vals + low_vals + close_vals) / 3.0
        r1 = 2 * pivot - low_vals
        s1 = 2 * pivot - high_vals
        r2 = pivot + (high_vals - low_vals)
        s2 = pivot - (high_vals - low_vals)
        r3 = high_vals + 2 * (pivot - low_vals)
        s3 = low_vals - 2 * (high_vals - pivot)
        return pivot, r1, s1, r2, s2, r3, s3
    
    pivot_1d, r1_1d, s1_1d, r2_1d, s2_1d, r3_1d, s3_1d = calculate_pivots(high_1d, low_1d, close_1d)
    
    # Calculate 6h volume 20-period average
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate RSI(14) on 6h close
    def rsi(values, period):
        delta = np.diff(values)
        seed = delta[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        if down == 0:
            rs = np.inf
        else:
            rs = up / down
        rsi_values = np.zeros_like(values)
        rsi_values[:period] = 100. - 100. / (1. + rs)
        for i in range(period, len(values)):
            delta = values[i] - values[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            if down == 0:
                rs = np.inf
            else:
                rs = up / down
            rsi_values[i] = 100. - 100. / (1. + rs)
        return rsi_values
    
    rsi_6h = rsi(close, 14)
    
    # Align all to primary timeframe (6h)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    vol_ma_20_6h_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_6h)  # using 1d as reference for alignment
    rsi_6h_aligned = align_htf_to_ltf(prices, df_1d, rsi_6h)  # using 1d as reference for alignment
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200 and pivots
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(vol_ma_20_6h_aligned[i]) or np.isnan(rsi_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20_6h_aligned[i]
        volume_low = volume[i] < 0.8 * vol_ma_20_6h_aligned[i]
        # Trend filter: EMA50 > EMA200 for uptrend, EMA50 < EMA200 for downtrend
        uptrend = ema_50_1w_aligned[i] > ema_200_1w_aligned[i]
        downtrend = ema_50_1w_aligned[i] < ema_200_1w_aligned[i]
        # RSI extremes
        rsi_overbought = rsi_6h_aligned[i] > 70
        rsi_oversold = rsi_6h_aligned[i] < 30
        
        if position == 0:
            # Long breakout: price breaks above 1d R1 with uptrend and volume
            if (close[i] > r1_1d_aligned[i] and 
                uptrend and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below 1d S1 with downtrend and volume
            elif (close[i] < s1_1d_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
            # Long mean reversion: price touches 1d S3 with oversold RSI and low volume
            elif (close[i] <= s3_1d_aligned[i] and 
                  rsi_oversold and 
                  volume_low):
                signals[i] = 0.25
                position = 1
            # Short mean reversion: price touches 1d R3 with overbought RSI and low volume
            elif (close[i] >= r3_1d_aligned[i] and 
                  rsi_overbought and 
                  volume_low):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 1d S1 (opposite side of pivot)
            if close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Or exit on mean reversion signal: price reaches 1d R3 with overbought RSI
            elif (close[i] >= r3_1d_aligned[i] and 
                  rsi_6h_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 1d R1 (opposite side of pivot)
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Or exit on mean reversion signal: price reaches 1d S3 with oversold RSI
            elif (close[i] <= s3_1d_aligned[i] and 
                  rsi_6h_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dPivot_R1S1_R3S3_Breakout_MeanRev_Volume_RSI"
timeframe = "6h"
leverage = 1.0