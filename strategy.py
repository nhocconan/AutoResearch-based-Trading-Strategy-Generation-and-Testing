#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_v1
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a trend filter that reduces whipsaws in ranging markets. Combined with RSI for momentum and volume confirmation, it captures trends while avoiding false signals in low-volatility periods. Designed for 1d timeframe to minimize trade frequency and maximize robustness across bull/bear regimes.
Target: 15-25 trades/year to stay well within fee drag limits.
"""

name = "1d_KAMA_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    close_w = df_w['close'].values
    # Calculate weekly EMA20 for trend filter
    ema20_w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_w_aligned = align_htf_to_ltf(prices, df_w, ema20_w)
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=1)  # 10-period volatility
    # Pad the beginning with NaN for proper alignment
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate volume moving average (20-period)
    vol_ma20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    # Get daily prices for signal generation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA20 (20), KAMA (10+30), RSI (14), volume MA (20)
    start_idx = max(20, 40, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema20_w_aligned[i]) or 
            np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs weekly EMA20
        uptrend_w = close[i] > ema20_w_aligned[i]
        downtrend_w = close[i] < ema20_w_aligned[i]
        
        # Price vs KAMA: price above KAMA = bullish momentum
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # RSI conditions: avoid extremes, look for momentum
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma20_aligned[i] * 1.5
        
        if position == 0:
            # Long entry: price above KAMA + uptrend + RSI bullish + volume
            if price_above_kama and uptrend_w and rsi_bullish and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA + downtrend + RSI bearish + volume
            elif price_below_kama and downtrend_w and rsi_bearish and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or trend fails or RSI overbought
            if price_below_kama or not uptrend_w or rsi_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or trend fails or RSI oversold
            if price_above_kama or not downtrend_w or rsi_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals