#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter
Hypothesis: On daily timeframe, use KAMA for trend direction (adaptive to choppy/trending markets),
combined with RSI for overbought/oversold conditions and Choppiness Index as regime filter.
Only trade when KAMA direction aligns with RSI extremes in non-choppy markets (CHOP < 38.2 for trend,
CHOP > 61.8 for mean reversion). Designed for low frequency (<25/year) to avoid fee drag.
Works in both bull (trend following) and bear (mean reversion in ranges) markets.
"""

name = "1d_KAMA_Direction_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA (Kaufman Adaptive Moving Average) ---
    # Fast EMA period = 2, Slow EMA period = 30
    fast_end = 2
    slow_end = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # Will be corrected below
    
    # Proper ER calculation: ER = |close[i] - close[i-n]| / sum(|close[i] - close[i-1]|) for i-n to i
    lookback = 10
    er = np.zeros(n)
    for i in range(lookback, n):
        if i >= lookback:
            price_change = np.abs(close[i] - close[i - lookback])
            price_volatility = np.sum(np.abs(np.diff(close[i - lookback:i + 1])))
            if price_volatility > 0:
                er[i] = price_change / price_volatility
            else:
                er[i] = 0
    # Fill beginning with 0
    er[:lookback] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_end + 1) - 2/(slow_end + 1)) + 2/(slow_end + 1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- RSI (14-period) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100  # When no losses
    
    # --- Choppiness Index (14-period) ---
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR (14-period) using Wilder's smoothing
    atr = np.zeros(n)
    atr[13] = np.mean(tr[1:15])
    for i in range(15, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of TR over 14 periods
    tr_sum = np.zeros(n)
    for i in range(13, n):
        if i == 13:
            tr_sum[i] = np.sum(tr[1:15])
        else:
            tr_sum[i] = tr_sum[i-1] - tr[i-14] + tr[i]
    
    # Choppiness Index
    chop = np.zeros(n)
    for i in range(13, n):
        if atr[i] > 0 and tr_sum[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (atr[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50  # Neutral
    
    # --- Weekly Trend Filter ---
    close_1w = df_1w['close'].values
    if len(close_1w) >= 20:
        sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
        sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
        weekly_uptrend = close > sma_20_1w_aligned
        weekly_downtrend = close < sma_20_1w_aligned
    else:
        weekly_uptrend = np.ones(n, dtype=bool)
        weekly_downtrend = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or invalid
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market regime
        is_trending = chop[i] < 38.2  # Trending market
        is_ranging = chop[i] > 61.8   # Ranging/choppy market
        
        # KAMA direction
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        if position == 0:
            if is_trending:
                # Trending market: follow KAMA direction with RSI filter
                if price_above_kama and rsi[i] < 40:  # Pullback in uptrend
                    signals[i] = 0.25
                    position = 1
                elif price_below_kama and rsi[i] > 60:  # Bounce in downtrend
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # Ranging market: mean reversion at RSI extremes
                if rsi[i] < 30:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70:  # Overbought
                    signals[i] = -0.25
                    position = -1
            # In neutral chop (38.2 <= CHOP <= 61.8), wait for clearer signal
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RSI overbought or trend change
                exit_signal = (rsi[i] > 70) or (price_below_kama and is_trending)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI oversold or trend change
                exit_signal = (rsi[i] < 30) or (price_above_kama and is_trending)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals