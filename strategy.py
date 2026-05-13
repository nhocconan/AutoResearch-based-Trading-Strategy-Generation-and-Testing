#!/usr/bin/env python3
"""
1d_Weekly_R1_S1_Breakout_With_Pullback
Hypothesis: Weekly pivot points (R1/S1) from weekly high/low/close act as strong support/resistance.
Breakouts above weekly R1 or below S1 with volume confirmation and daily trend alignment capture momentum moves.
After breakout, wait for a pullback to the breakout level (R1/S1) or EMA20 before entering, reducing false breakouts.
Exit on reversion to weekly pivot point (PP) or trend reversal. Position size 0.25 targets ~15-25 trades/year.
Works in both bull (breakouts with trend) and bear (mean reversion at extremes) markets via trend filter.
"""

name = "1d_Weekly_R1_S1_Breakout_With_Pullback"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    weekly_pp = (h_1w + l_1w + c_1w) / 3.0
    weekly_r1 = c_1w + (h_1w - l_1w) * 1.1 / 12.0
    weekly_s1 = c_1w - (h_1w - l_1w) * 1.1 / 12.0
    
    # Align weekly pivots to daily chart (wait for weekly close)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Daily trend filter: EMA50
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # EMA20 for pullback entry
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    breakout_level = 0.0  # tracks breakout level for pullback entry
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # Check for potential breakout
            bullish_breakout = (close[i] > weekly_r1_aligned[i]) and volume_filter[i]
            bearish_breakout = (close[i] < weekly_s1_aligned[i]) and volume_filter[i]
            
            if bullish_breakout:
                # Set breakout level and wait for pullback
                breakout_level = weekly_r1_aligned[i]
                # Enter on pullback to breakout level or EMA20 with uptrend
                if (close[i] <= breakout_level * 1.005 or close[i] <= ema20[i]) and close[i] > ema50[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_breakout:
                # Set breakout level and wait for pullback
                breakout_level = weekly_s1_aligned[i]
                # Enter on pullback to breakout level or EMA20 with downtrend
                if (close[i] >= breakout_level * 0.995 or close[i] >= ema20[i]) and close[i] < ema50[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly pivot or trend reverses
            if (close[i] < weekly_pp_aligned[i]) or \
               (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly pivot or trend reverses
            if (close[i] > weekly_pp_aligned[i]) or \
               (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals