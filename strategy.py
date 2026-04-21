#!/usr/bin/env python3
"""
4h_1d_Choppy_Trend_MeanRev_v1
Hypothesis: Trend following in trending markets (ADX>25, EMA200), mean reversion in choppy markets (ADX<20, Bollinger Bands). Uses 1d trend filter and 4h ADX for regime detection. Combines 4h EMA crossover with Bollinger mean reversion. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX(14) for regime detection
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * dm_plus_smooth / (atr + 1e-10)
    minus_di = 100 * dm_minus_smooth / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # EMA crossover (8/21) for trend following
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Bollinger Bands (20,2) for mean reversion
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(ema8[i]) or np.isnan(ema21[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        trend_up = ema200_1d_aligned[i] > ema200_1d_aligned[i-1]
        
        if position == 0:
            # Trending market: ADX > 25
            if adx[i] > 25:
                # EMA crossover with trend filter
                if ema8[i] > ema21[i] and trend_up:
                    signals[i] = 0.25
                    position = 1
                elif ema8[i] < ema21[i] and not trend_up:
                    signals[i] = -0.25
                    position = -1
            # Choppy market: ADX < 20
            elif adx[i] < 20:
                # Mean reversion at Bollinger Bands
                if price <= lower[i]:
                    signals[i] = 0.25
                    position = 1
                elif price >= upper[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: EMA cross down OR mean reversion to middle
            if ema8[i] < ema21[i] or price >= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: EMA cross up OR mean reversion to middle
            if ema8[i] > ema21[i] or price <= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Choppy_Trend_MeanRev_v1"
timeframe = "4h"
leverage = 1.0