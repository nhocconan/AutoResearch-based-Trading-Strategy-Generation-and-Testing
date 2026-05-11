#!/usr/bin/env python3
"""
4h_1w_EquityCurveTrend_With_Volume
Hypothesis: Use 1-week equity curve of a simple 4h SMA crossover to filter trend, 
and enter on 4h Donchian breakouts with volume confirmation. 
In weekly uptrend (equity curve rising), go long on 4h breakouts above upper channel. 
In weekly downtrend (equity curve falling), go short on breakouts below lower channel. 
Exit on opposite Donchian breakout. This avoids whipsaws by using weekly trend strength 
while keeping trade frequency low (~20-40/year) to minimize fee drag.
Works in bull by buying dips in uptrend; works in bear by selling rallies in downtrend.
"""

name = "4h_1w_EquityCurveTrend_With_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-week data for equity curve calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly equity curve from simple 4h SMA crossover ---
    # Calculate 4h SMA(20) and SMA(50) on weekly timeframe using 4h close prices
    # We need to resample 4h close to weekly for proper SMA calculation
    # But instead, we'll calculate the equity curve by simulating 4h strategy on weekly bars
    # Simple approach: use weekly close to calculate SMA crossover equity
    
    close_1w = df_1w['close'].values
    
    # Calculate SMA(20) and SMA(50) on weekly closes
    sma20_1w = np.full(len(close_1w), np.nan)
    sma50_1w = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if i >= 19:
            sma20_1w[i] = np.mean(close_1w[i-19:i+1])
        if i >= 49:
            sma50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    # Calculate equity curve: cumulative returns from SMA crossover strategy
    # Long when SMA20 > SMA50, short when SMA20 < SMA50
    position_1w = np.zeros(len(close_1w))
    equity_1w = np.ones(len(close_1w))  # Start at 1 (100%)
    
    for i in range(1, len(close_1w)):
        if not np.isnan(sma20_1w[i]) and not np.isnan(sma50_1w[i]):
            if sma20_1w[i] > sma50_1w[i]:
                position_1w[i] = 1   # Long
            elif sma20_1w[i] < sma50_1w[i]:
                position_1w[i] = -1  # Short
        
        # Calculate return from previous close to current close
        if i > 0 and not np.isnan(close_1w[i-1]) and not np.isnan(close_1w[i]):
            ret = (close_1w[i] - close_1w[i-1]) / close_1w[i-1]
            equity_1w[i] = equity_1w[i-1] * (1 + position_1w[i-1] * ret)
        else:
            equity_1w[i] = equity_1w[i-1]
    
    # Weekly trend: equity curve rising or falling
    # Use 3-period slope of equity curve
    equity_slope_1w = np.full(len(equity_1w), np.nan)
    for i in range(3, len(equity_1w)):
        equity_slope_1w[i] = (equity_1w[i] - equity_1w[i-3]) / 3
    
    # Align weekly equity slope to 4h timeframe
    equity_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, equity_slope_1w)
    
    # --- 4h Donchian Channel (20-period) for entry ---
    donch_high_4h = np.full(n, np.nan)
    donch_low_4h = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high_4h[i] = np.max(high[i-20:i])
        donch_low_4h[i] = np.min(low[i-20:i])
    
    # Volume confirmation: volume > 1.5x average volume
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(equity_slope_1w_aligned[i]) or np.isnan(donch_high_4h[i]) or 
            np.isnan(donch_low_4h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend from equity curve slope
        weekly_uptrend = equity_slope_1w_aligned[i] > 0
        weekly_downtrend = equity_slope_1w_aligned[i] < 0
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend with volume confirmation
            if weekly_uptrend and vol_confirmed and close[i] > donch_high_4h[i]:
                # Long: weekly uptrend + volume + breakout above upper channel
                signals[i] = 0.25
                position = 1
            elif weekly_downtrend and vol_confirmed and close[i] < donch_low_4h[i]:
                # Short: weekly downtrend + volume + breakdown below lower channel
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout
            if position == 1:
                # Exit long: breakdown below lower Donchian channel
                if close[i] < donch_low_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: breakout above upper Donchian channel
                if close[i] > donch_high_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals