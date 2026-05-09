#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_MeanReversion
# Hypothesis: KAMA trend direction combined with RSI mean-reversion on daily timeframe.
# Long when KAMA trend up and RSI < 30 (oversold), short when KAMA trend down and RSI > 70 (overbought).
# Uses 1-week trend filter to avoid counter-trend trades. Designed for low trade frequency (<25/year) to minimize fee drag.

name = "1d_KAMA_Trend_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (ER=10, fast=2, slow=30) on 1d
    def kama(close, er_period=10, fast=2, slow=30):
        n = len(close)
        kama = np.full(n, np.nan)
        if n < er_period + 1:
            return kama
        # Efficiency Ratio
        change = np.abs(np.diff(close, lag=er_period))
        abs_change = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.diff(close), 'shape') else None
        # Simplified ER calculation for 1D array
        er = np.zeros(n)
        for i in range(er_period, n):
            if i >= er_period:
                diff = np.abs(close[i] - close[i-er_period])
                total = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
                if total != 0:
                    er[i] = diff / total
                else:
                    er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
        # Initialize kama
        kama[er_period] = close[er_period]
        for i in range(er_period+1, n):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
        return kama
    
    kama_1d = kama(close_1d, er_period=10, fast=2, slow=30)
    
    # Calculate RSI(14) on 1d
    def rsi(close, period=14):
        n = len(close)
        rsi = np.full(n, np.nan)
        if n < period + 1:
            return rsi
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = rsi(close_1d, period=14)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Simple trend: price above/below 20-period EMA on weekly
    ema20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema20_1w[i] = (close_1w[i] * 2 + ema20_1w[i-1] * 18) / 20
    
    # Align indicators to 1d timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(11, 15, 20)  # KAMA needs ~11, RSI needs 15, weekly EMA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine KAMA trend (price above/below KAMA)
        kama_trend_up = close[i] > kama_1d_aligned[i]
        
        # Determine weekly trend filter
        weekly_trend_up = close[i] > ema20_1w_aligned[i]
        
        if position == 0:
            # Enter long: KAMA trend up, RSI oversold (<30), and weekly trend up
            if kama_trend_up and rsi_1d_aligned[i] < 30 and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA trend down, RSI overbought (>70), and weekly trend down
            elif not kama_trend_up and rsi_1d_aligned[i] > 70 and not weekly_trend_up:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA trend turns down or RSI overbought (>70)
            if not kama_trend_up or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA trend turns up or RSI oversold (<30)
            if kama_trend_up or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals