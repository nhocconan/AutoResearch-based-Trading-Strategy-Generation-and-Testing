#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, combined with RSI(14) for momentum and Choppiness Index(14) for regime filtering. Only trade when KAMA slope confirms trend, RSI is not extreme (avoiding exhaustion), and market is trending (CHOP < 38.2) or mean-reverting (CHOP > 61.8) with appropriate RSI bias. This avoids choppy whipsaws and captures strong trends while fading extremes in ranging markets. Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag and work in both bull and bear markets.
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
    
    # KAMA parameters
    fast_sc = 0.666  # 2/(2+1)
    slow_sc = 0.0645 # 2/(30+1)
    
    # Calculate Efficiency Ratio (ER) and smoothed alpha for KAMA
    change = np.abs(np.diff(close, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, 1)), axis=0)  # 10-period sum of abs changes
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    # Choppiness Index(14)
    atr_tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_tr[0] = high[0] - low[0]
    atr_sum = pd.Series(atr_tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((highest_high - lowest_low) != 0,
                    100 * np.log10(atr_sum / np.log(14) / (highest_high - lowest_low)),
                    50)  # neutral if range is zero
    
    # Get 1w data for HTF trend filter (optional reinforcement)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 20:
        # Weekly EMA(20) for major trend
        ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
        ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    else:
        ema_20_1w_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of KAMA seed(10), RSI(14), CHOP(14), weekly EMA(20)
    start_idx = max(10, 14, 14, 20) + 5  # extra buffer
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            (i < len(ema_20_1w_aligned) and np.isnan(ema_20_1w_aligned[i]))):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        kama_val = kama[i]
        kama_prev = kama[i-1]
        close_val = close[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_20_1w_val = ema_20_1w_aligned[i] if i < len(ema_20_1w_aligned) else np.nan
        
        # KAMA slope: rising if current > previous
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        # Regime filters
        chop_trending = chop_val < 38.2   # trending market
        chop_ranging = chop_val > 61.8    # ranging market
        
        # RSI filters: avoid extremes, look for momentum
        rsi_overbought = rsi_val > 70
        rsi_oversold = rsi_val < 30
        rsi_momentum_up = 50 < rsi_val < 70
        rsi_momentum_down = 30 < rsi_val < 50
        
        if position == 0:
            # Long conditions:
            # 1. In trending market: KAMA rising + RSI momentum up
            # 2. In ranging market: RSI oversold bounce (mean reversion)
            long_trending = kama_rising and chop_trending and rsi_momentum_up
            long_ranging = (not chop_trending and not chop_ranging) and rsi_oversold and kama_rising
            # Optional: weekly trend alignment
            weekly_uptrend = not np.isnan(ema_20_1w_val) and close_val > ema_20_1w_val
            
            long_signal = (long_trending or long_ranging) and weekly_uptrend
            
            # Short conditions:
            # 1. In trending market: KAMA falling + RSI momentum down
            # 2. In ranging market: RSI overbought fade (mean reversion)
            short_trending = kama_falling and chop_trending and rsi_momentum_down
            short_ranging = (not chop_trending and not chop_ranging) and rsi_overbought and kama_falling
            
            short_signal = (short_trending or short_ranging) and (not weekly_uptrend or np.isnan(ema_20_1w_val))
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA turns down OR RSI overbought in trending market
            if kama_falling or (chop_trending and rsi_overbought):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA turns up OR RSI oversold in trending market
            if kama_rising or (chop_trending and rsi_oversold):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0