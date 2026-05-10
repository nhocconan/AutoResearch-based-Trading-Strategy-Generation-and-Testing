#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_MeanReversion_1dFilter
# Hypothesis: KAMA on 12h identifies adaptive trend direction, RSI(14) on 12h provides mean-reversion entries
# when price deviates from trend, filtered by 1d ADX to avoid ranging markets. Designed for low trade
# frequency (<20/year) to minimize fee drag and work in both bull and bear markets via trend adaptation.
# Target: 15-25 trades/year on 12h timeframe.

name = "12h_KAMA_Trend_RSI_MeanReversion_1dFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_fast=2, er_slow=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def adx(high, low, close, period=14):
    """Average Directional Index"""
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength filter
    adx_1d = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 12h data for signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA(10,2,30) for trend
    kama_val = kama(close, 2, 30)
    
    # RSI(14) for mean reversion
    rsi_val = rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (30) + RSI (14) + ADX (14)
    start_idx = max(30, 14)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_val[i]) or 
            np.isnan(rsi_val[i]) or
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when ADX > 25 (trending market)
        if adx_1d_aligned[i] > 25:
            if position == 0:
                # Long: price below KAMA (pullback in uptrend) + RSI oversold
                if close[i] < kama_val[i] and rsi_val[i] < 30:
                    signals[i] = 0.25
                    position = 1
                # Short: price above KAMA (pullback in downtrend) + RSI overbought
                elif close[i] > kama_val[i] and rsi_val[i] > 70:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Long exit: price crosses above KAMA or RSI overbought
                if close[i] > kama_val[i] or rsi_val[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price crosses below KAMA or RSI oversold
                if close[i] < kama_val[i] or rsi_val[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In ranging markets (ADX <= 25), stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals