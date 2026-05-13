# 165107
#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Price_Channel_Breakout_Trend
Hypothesis: Weekly pivot points (PP, R1, S1) act as strong support/resistance. A breakout above weekly R1 or below S1 with price channel (Donchian 20) confirmation and aligned weekly trend (price above/below weekly EMA50) signals continuation. Weekly timeframe reduces noise, price channel filters false breakouts, and trend alignment ensures momentum. Designed for low trade frequency (~20-40/year) on 6h bars to minimize fee drag while capturing sustained moves in both bull and bear regimes.
"""

name = "6h_Weekly_Pivot_Price_Channel_Breakout_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for pivots, trend, and price channel (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Weekly pivot points from previous week
    # Pivot Point = (H + L + C) / 3
    pp = (df_w['high'] + df_w['low'] + df_w['close']) / 3
    # Range = H - L
    range_w = df_w['high'] - df_w['low']
    # Weekly R1 = (2 * PP) - L
    weekly_r1 = (2 * pp) - df_w['low']
    # Weekly S1 = (2 * PP) - H
    weekly_s1 = (2 * pp) - df_w['high']
    
    # Align weekly pivots to 6h (use previous week's values)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_w, weekly_r1.values)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_w, weekly_s1.values)
    
    # Weekly trend filter: EMA(50) on close
    ema50_w = pd.Series(df_w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    
    # Price channel: Donchian(20) on 6h high/low for breakout confirmation
    # Upper channel = max(high, lookback=20)
    # Lower channel = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        if position == 0:
            # LONG: Price breaks above weekly R1, above Donchian upper, and above weekly EMA50 (uptrend)
            if (close[i] > weekly_r1_aligned[i] and 
                close[i] > donchian_upper[i] and 
                close[i] > ema50_w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S1, below Donchian lower, and below weekly EMA50 (downtrend)
            elif (close[i] < weekly_s1_aligned[i] and 
                  close[i] < donchian_lower[i] and 
                  close[i] < ema50_w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below weekly R1 (failed breakout) OR below Donchian upper
            if (close[i] < weekly_r1_aligned[i] or 
                close[i] < donchian_upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above weekly S1 (failed breakdown) OR above Donchian lower
            if (close[i] > weekly_s1_aligned[i] or 
                close[i] > donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals