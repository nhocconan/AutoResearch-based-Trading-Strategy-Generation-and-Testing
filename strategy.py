#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume
Hypothesis: On daily timeframe, Camarilla R1/S1 breakouts with volume confirmation and weekly trend filter capture institutional order flow. Weekly trend (price > EMA50) ensures alignment with higher timeframe momentum, reducing false signals in ranging markets. Designed for low trade frequency (15-25/year) to minimize fee drag while capturing strong directional moves in both bull and bear markets.
"""

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Camarilla levels from previous day
    # R1 = close + 1.12 * (high - low) / 12
    # S1 = close - 1.12 * (high - low) / 12
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Handle first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_width = (prev_high - prev_low) * 1.12 / 12
    r1 = prev_close + camarilla_width
    s1 = prev_close - camarilla_width
    
    # Volume confirmation: current volume > 1.8x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from second bar (need previous day)
        if position == 0:
            # LONG: Close breaks above R1, volume confirmation, price above weekly EMA50 (uptrend)
            if (close[i] > r1[i] and 
                volume_filter[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1, volume confirmation, price below weekly EMA50 (downtrend)
            elif (close[i] < s1[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close breaks below S1 (reversal) OR weekly trend turns bearish
            if (close[i] < s1[i]) or (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close breaks above R1 (reversal) OR weekly trend turns bullish
            if (close[i] > r1[i]) or (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals