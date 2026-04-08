#!/usr/bin/env python3
"""
4x4 Grid Strategy for BTC/ETH/SOL USDT-M Perpetual Futures
Hypothesis: The 4x4 grid strategy combines 4 independent signals across 4 timeframes (4h, 1d, 1w, and a volatility regime filter) to generate high-probability trades with low frequency. Each signal must align for entry, reducing false signals and overtrading. The strategy is designed to work in both bull and bear markets by using trend-following on higher timeframes and volatility-based regime filtering to avoid choppy markets. Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4x4_grid_strategy"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Signal 1: 4h EMA(21) vs EMA(50) trend
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h = ema_21 - ema_50  # >0 = uptrend, <0 = downtrend
    
    # Signal 2: 1d Donchian breakout (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Signal 3: 1w EMA(50) trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Signal 4: Volatility regime filter (ATR ratio)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = high[0] - close[0]
    tr3[0] = low[0] - close[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # <1 = low volatility, >1 = high volatility
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_4h[i]) or 
            np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: 4h trend turns negative OR price breaks below 1d Donchian low
            if trend_4h[i] < 0 or close[i] < donchian_low_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: 4h trend turns positive OR price breaks above 1d Donchian high
            if trend_4h[i] > 0 or close[i] > donchian_high_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # All 4 signals must agree for entry
            # Signal 1: 4h EMA trend
            s1_long = trend_4h[i] > 0
            s1_short = trend_4h[i] < 0
            
            # Signal 2: Price vs 1d Donchian channels
            s2_long = close[i] > donchian_high_1d_aligned[i]
            s2_short = close[i] < donchian_low_1d_aligned[i]
            
            # Signal 3: Price vs 1w EMA50 trend
            s3_long = close[i] > ema_50_1w_aligned[i]
            s3_short = close[i] < ema_50_1w_aligned[i]
            
            # Signal 4: Volatility regime (prefer low volatility for breakouts)
            s4 = atr_ratio[i] < 1.0  # Low volatility regime
            
            # Long entry: all signals agree on long + low volatility
            if s1_long and s2_long and s3_long and s4:
                position = 1
                signals[i] = 0.25
            # Short entry: all signals agree on short + low volatility
            elif s1_short and s2_short and s3_short and s4:
                position = -1
                signals[i] = -0.25
    
    return signals