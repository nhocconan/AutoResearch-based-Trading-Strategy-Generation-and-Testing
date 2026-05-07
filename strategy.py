#!/usr/bin/env python3
# 1d_KAMA_RSI_ChopFilter_Trend
# Hypothesis: 1d chart strategy using KAMA trend direction confirmed by RSI and filtered by Choppiness Index.
# Long when KAMA > previous KAMA, RSI > 50, and Choppiness Index > 61.8 (range market).
# Short when KAMA < previous KAMA, RSI < 50, and Choppiness Index > 61.8 (range market).
# Uses weekly trend filter: only trade in direction of weekly EMA34 trend.
# Designed to work in both bull and bear markets by combining trend-following (KAMA) with mean-reversion (RSI) in ranging conditions.
# Target: 15-25 trades/year per symbol to minimize fee drag while maintaining edge.

timeframe = "1d"
name = "1d_KAMA_RSI_ChopFilter_Trend"
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
    volume = prices['volume'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation
    price_change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility_sum = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    
    # Simplified: use pandas for ER calculation
    close_series = pd.Series(close)
    price_change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=1).sum()
    ER = price_change / volatility.replace(0, np.nan)
    ER = ER.fillna(0).values
    
    # Smoothing constants
    fastest_SC = 2 / (2 + 1)   # EMA(2)
    slowest_SC = 2 / (30 + 1)  # EMA(30)
    SC = (ER * (fastest_SC - slowest_SC) + slowest_SC) ** 2
    SC = np.nan_to_num(SC, nan=0.0)
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + SC[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).fillna(50).values
    
    # Calculate Choppiness Index (14-period)
    def true_range(high, low, close_prev):
        return np.maximum(
            np.maximum(high - low, np.abs(high - close_prev)),
            np.abs(low - close_prev)
        )
    
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    tr = true_range(high, low, close_prev)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.where(
        (atr14 > 0) & (highest_high - lowest_low > 0),
        100 * np.log10(atr14 / (highest_high - lowest_low)) / np.log10(14),
        50  # neutral when undefined
    )
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend direction
    weekly_close = df_1w['close'].values
    weekly_ema34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14, 10)  # Ensure we have all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i-1]) if i>0 else False or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(weekly_ema34_aligned[i]) or
            np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if we are in a ranging market (chop > 61.8)
        is_ranging = chop[i] > 61.8
        
        if position == 0:
            # Long conditions: KAMA up, RSI > 50, ranging market, and price above weekly EMA34 (bullish trend)
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and 
                is_ranging and
                close[i] > weekly_ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA down, RSI < 50, ranging market, and price below weekly EMA34 (bearish trend)
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and 
                  is_ranging and
                  close[i] < weekly_ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA turns down OR RSI < 40 (overbought exit)
            if (kama[i] < kama[i-1] or rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA turns up OR RSI > 60 (oversold exit)
            if (kama[i] > kama[i-1] or rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals