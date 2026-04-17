#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop_Filter
Strategy: KAMA trend direction + RSI momentum + Choppiness regime filter.
Long: KAMA rising, RSI > 50, Choppiness Index < 50 (trending market)
Short: KAMA falling, RSI < 50, Choppiness Index < 50
Exit: Opposite KAMA direction
Position size: 0.30
Designed to capture trend momentum while avoiding choppy markets.
Works in both bull (trend following) and bear (trend following shorts).
Timeframe: 1d
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[-30:])  # 30-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14-period)
    # True Range
    tr1 = np.abs(np.subtract(prices['high'], prices['low']))
    tr2 = np.abs(np.subtract(prices['high'], np.roll(prices['close'], 1)))
    tr3 = np.abs(np.subtract(prices['low'], np.roll(prices['close'], 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(prices['high']).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(prices['low']).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    ci = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    ci = np.where((hh - ll) != 0, ci, 50)  # avoid division by zero
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ci[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI momentum
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Choppiness filter: trending market (CI < 50)
        trending_market = ci[i] < 50
        
        # Weekly trend filter
        price_above_weekly_ema = close[i] > ema34_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: KAMA rising + RSI bullish + trending market + above weekly EMA
            if kama_rising and rsi_bullish and trending_market and price_above_weekly_ema:
                signals[i] = 0.30
                position = 1
            # Short: KAMA falling + RSI bearish + trending market + below weekly EMA
            elif kama_falling and rsi_bearish and trending_market and price_below_weekly_ema:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns flat or falling
            if not kama_rising:  # KAMA flat or falling
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: KAMA turns flat or rising
            if not kama_falling:  # KAMA flat or rising
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0