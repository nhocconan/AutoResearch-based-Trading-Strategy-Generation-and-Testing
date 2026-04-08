#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Hypothesis: On daily timeframe, use KAMA for trend direction, RSI for momentum/overbought-oversold,
# and Choppiness Index for regime filtering. Enter long when KAMA trends up, RSI < 30 (oversold),
# and market is trending (CHOP < 38.2). Enter short when KAMA trends down, RSI > 70 (overbought),
# and market is trending. Uses weekly trend filter to avoid counter-trend trades.
# Designed for 15-25 trades/year on 1d to minimize fee drag while capturing meaningful moves.
# Works in bull markets via trend-following longs and bear markets via trend-following shorts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 1 else np.abs(change)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index: higher = ranging, lower = trending"""
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    chop = np.zeros_like(close)
    for i in range(len(close)):
        if atr[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(atr[i] * period / (max_high[i] - min_low[i])) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral when undefined
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # KAMA for trend direction
    kama = calculate_kama(close, er_period=10, fast=2, slow=30)
    
    # RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index for regime
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Weekly trend filter: EMA(21) on weekly close
    df_1w = get_htf_data(prices, '1w')
    ema21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema21_1w_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        kama_slope = kama[i] - kama[i-1]  # positive = rising trend
        chop_value = chop[i]
        rsi_value = rsi[i]
        weekly_trend = ema21_1w_aligned[i]  # above = bullish week, below = bearish week
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR chop becomes too high (ranging) OR weekly trend turns bearish
            if kama_slope < 0 or chop_value > 61.8 or close[i] < weekly_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR chop becomes too high (ranging) OR weekly trend turns bullish
            if kama_slope > 0 or chop_value > 61.8 or close[i] > weekly_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: KAMA rising, RSI oversold, chop low (trending), weekly trend bullish
            if (kama_slope > 0 and 
                rsi_value < 30 and 
                chop_value < 38.2 and 
                close[i] > weekly_trend):
                position = 1
                signals[i] = 0.25
            # Short entry: KAMA falling, RSI overbought, chop low (trending), weekly trend bearish
            elif (kama_slope < 0 and 
                  rsi_value > 70 and 
                  chop_value < 38.2 and 
                  close[i] < weekly_trend):
                position = -1
                signals[i] = -0.25
    
    return signals