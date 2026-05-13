#!/usr/bin/env python3
"""
4h_Three_Month_Pivot_Mean_Reversion
Hypothesis: Price tends to revert to 3-month (13-week) pivot levels derived from monthly high/low/close. 
Combines monthly pivot calculation with 1-week trend filter and volume confirmation to capture 
mean-reversion moves in both bull and bear markets. Designed for low trade frequency (15-30/year).
"""

name = "4h_Three_Month_Pivot_Mean_Reversion"
timeframe = "4h"
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
    
    # Calculate 13-week (quarterly) pivot levels from monthly data
    # Get 1-month data for pivot calculation
    df_1m = get_htf_data(prices, '1M')
    if len(df_1m) < 3:
        return np.zeros(n)
    
    # Calculate monthly pivot: P = (H + L + C) / 3
    monthly_high = df_1m['high'].values
    monthly_low = df_1m['low'].values
    monthly_close = df_1m['close'].values
    monthly_pivot = (monthly_high + monthly_low + monthly_close) / 3.0
    
    # Calculate support and resistance levels
    monthly_r1 = 2 * monthly_pivot - monthly_low
    monthly_s1 = 2 * monthly_pivot - monthly_high
    monthly_r2 = monthly_pivot + (monthly_high - monthly_low)
    monthly_s2 = monthly_pivot - (monthly_high - monthly_low)
    
    # Align monthly pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1m, monthly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1m, monthly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1m, monthly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1m, monthly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1m, monthly_s2)
    
    # 1-week trend filter (EMA 20)
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price near support with bullish 1w trend and volume
            if (close[i] <= s1_aligned[i] * 1.02 and  # Within 2% of S1
                close[i] > ema_20_1w_aligned[i] and   # Above 1w EMA (bullish bias)
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near resistance with bearish 1w trend and volume
            elif (close[i] >= r1_aligned[i] * 0.98 and  # Within 2% of R1
                  close[i] < ema_20_1w_aligned[i] and   # Below 1w EMA (bearish bias)
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot or 1w trend turns bearish
            if (close[i] >= pivot_aligned[i] * 0.995 or  # Near pivot
                close[i] < ema_20_1w_aligned[i]):        # Below 1w EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot or 1w trend turns bullish
            if (close[i] <= pivot_aligned[i] * 1.005 or  # Near pivot
                close[i] > ema_20_1w_aligned[i]):        # Above 1w EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals