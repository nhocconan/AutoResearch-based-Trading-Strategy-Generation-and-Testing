#!/usr/bin/env python3
"""
4h_1d_volatility_breakout_v1
Hypothesis: Breakouts from volatility contraction (low Bollinger Bandwidth) followed by expansion, filtered by 1d trend (EMA50) and volume confirmation. Works in bull/bear by aligning with higher timeframe trend and using volatility as entry signal.
Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag.
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
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) for volatility squeeze
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bandwidth = (upper - lower) / sma20  # Normalized bandwidth
    
    # Bollinger Bandwidth percentile (20-period lookback) to detect squeeze
    bandwidth_series = pd.Series(bandwidth)
    bandwidth_percentile = bandwidth_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volatility squeeze: bandwidth < 20th percentile
    squeeze = bandwidth_percentile < 20
    
    # Volatility expansion: bandwidth > 50th percentile AND increasing
    bandwidth_increasing = bandwidth > np.roll(bandwidth, 1)
    expansion = (bandwidth_percentile > 50) & bandwidth_increasing
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(bandwidth_percentile[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: volatility expansion after squeeze, price above 1d EMA50, volume confirmation
        if (squeeze[i-1] and expansion[i] and close[i] > ema50_1d_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: volatility expansion after squeeze, price below 1d EMA50, volume confirmation
        elif (squeeze[i-1] and expansion[i] and close[i] < ema50_1d_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: volatility contraction returns (bandwidth < 30th percentile) or opposite volatility expansion
        elif position == 1 and bandwidth_percentile[i] < 30:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bandwidth_percentile[i] < 30:
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

name = "4h_1d_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0