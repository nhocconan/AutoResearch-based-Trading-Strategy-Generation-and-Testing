#!/usr/bin/env python3
"""
1d_1w_Stochastic_BullBear
Hypothesis: In bull markets, buy when weekly stochastic is oversold (<20) and price pulls back to daily EMA20.
In bear markets, sell when weekly stochastic is overbought (>80) and price rallies to daily EMA20.
Use daily ATR for stop/reverse. Stochastic filters avoid counter-trend trades in strong trends.
Designed for low trade frequency (<20/year) to avoid fee drag. Works in both bull (buy dips) and bear (sell rallies).
"""

name = "1d_1w_Stochastic_BullBear"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for stochastic
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Stochastic (14,3,3) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    lowest_low = np.full(len(close_1w), np.nan)
    highest_high = np.full(len(close_1w), np.nan)
    for i in range(13, len(close_1w)):
        lowest_low[i] = np.min(low_1w[i-13:i+1])
        highest_high[i] = np.max(high_1w[i-13:i+1])
    
    stoch_k = np.full(len(close_1w), np.nan)
    for i in range(13, len(close_1w)):
        if highest_high[i] - lowest_low[i] != 0:
            stoch_k[i] = (close_1w[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i]) * 100
        else:
            stoch_k[i] = 50.0
    
    # %D = SMA of %K, period 3
    stoch_d = np.full(len(close_1w), np.nan)
    for i in range(15, len(close_1w)):  # 13 + 2 for 3-period SMA
        stoch_d[i] = np.mean(stoch_k[i-2:i+1])
    
    # --- Daily EMA20 for trend/pullback ---
    ema20 = np.full(n, np.nan)
    for i in range(20, n):
        if i == 20:
            ema20[i] = np.mean(close[0:20])
        else:
            ema20[i] = (close[i] * 2 / (20 + 1)) + (ema20[i-1] * (19 / (20 + 1)))
    
    # --- Daily ATR(14) for volatility ---
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[0:14])
        else:
            atr[i] = (tr[i] * 1 / 14) + (atr[i-1] * 13 / 14)
    
    # Align weekly indicators to daily
    stoch_d_aligned = align_htf_to_ltf(prices, df_1w, stoch_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(weekly stoch needs 15 bar, EMA20, ATR14)
    start_idx = max(15, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(stoch_d_aligned[i]) or
            np.isnan(ema20[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bull/Bear regime from weekly stochastic
        bull_regime = stoch_d_aligned[i] < 50  # Below 50 = bearish bias? Actually, <50 is bearish, >50 bullish
        # Let's reverse: >50 = bullish bias, <50 = bearish bias
        bull_regime = stoch_d_aligned[i] > 50
        bear_regime = stoch_d_aligned[i] < 50
        
        # Price relative to daily EMA20
        price_near_ema = np.abs(close[i] - ema20[i]) < (0.5 * atr[i])  # Within 0.5 ATR of EMA20
        
        if position == 0:
            # In bull regime, buy when price pulls back to EMA20 (dip buying)
            if bull_regime and price_near_ema and close[i] <= ema20[i]:
                signals[i] = 0.25
                position = 1
            # In bear regime, sell when price rallies to EMA20 (sell the rally)
            elif bear_regime and price_near_ema and close[i] >= ema20[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price moves above EMA20 or weekly turns bearish
                if close[i] > ema20[i] or stoch_d_aligned[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price moves below EMA20 or weekly turns bullish
                if close[i] < ema20[i] or stoch_d_aligned[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals