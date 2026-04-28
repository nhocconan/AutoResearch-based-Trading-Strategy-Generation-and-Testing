#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop_Filter
Hypothesis: Daily KAMA trend direction with RSI momentum and Choppiness index regime filter.
KAMA adapts to market noise, reducing false signals in choppy conditions. RSI filters extremes,
and Choppiness index determines market regime (trending vs ranging) to apply appropriate logic.
Designed for low turnover (10-30 trades/year) to minimize fee impact and work in both bull/bear markets.
"""

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
    
    # Get weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    for i in range(1, len(change)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smooth ER over 10 periods
    er_smooth = pd.Series(er).ewm(span=10, adjust=False, min_periods=1).mean().values
    # SC = [ER * (fastest SC - slowest SC) + slowest SC]^2
    fastest_sc = 2 / (2 + 1)   # EMA(2)
    slowest_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er_smooth * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14-period)
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.zeros_like(close)
    for i in range(13, len(close)):
        sum_atr = np.sum(atr[i-13:i+1])
        range_hl = max_high[i] - min_low[i]
        if range_hl > 0:
            chop[i] = 100 * np.log10(sum_atr / range_hl) / np.log10(14)
        else:
            chop[i] = 50  # Neutral when no range
    
    # Align weekly EMA20 to daily
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Conditions
    # Trend filter: price > weekly EMA20 = bullish, < = bearish
    trend_up = close > ema_20_1w_aligned
    trend_down = close < ema_20_1w_aligned
    
    # KAMA direction: price above KAMA = bullish momentum, below = bearish
    kama_up = close > kama
    kama_down = close < kama
    
    # RSI filters: avoid extremes, look for momentum
    rsi_not_overbought = rsi < 70
    rsi_not_oversold = rsi > 30
    rsi_bullish = rsi > 50
    rsi_bearish = rsi < 50
    
    # Choppiness regime: < 38.2 = trending, > 61.8 = ranging
    chopping = chop > 61.8
    trending = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # In trending regime: follow KAMA + RSI momentum with weekly trend filter
        # In choppy regime: mean revert at RSI extremes
        
        if trending[i]:
            # Trending market: follow momentum with trend filter
            long_entry = kama_up[i] and rsi_bullish[i] and trend_up[i] and rsi_not_overbought[i]
            short_entry = kama_down[i] and rsi_bearish[i] and trend_down[i] and rsi_not_oversold[i]
        else:
            # Choppy/ranging market: mean reversion at RSI extremes
            long_entry = rsi_not_oversold[i] and rsi < 35  # Oversold bounce
            short_entry = rsi_not_overbought[i] and rsi > 65  # Overbought reversal
        
        # Exit conditions
        long_exit = (not kama_up[i]) or (rsi > 70) or (trend_down[i] and trending[i])
        short_exit = (not kama_down[i]) or (rsi < 30) or (trend_up[i] and trending[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0