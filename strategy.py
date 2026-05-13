#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume
Hypothesis: Daily Camarilla pivot breakouts with weekly trend filter and volume confirmation capture institutional interest in both bull and bear markets. The Camarilla R1/S1 levels act as key intraday support/resistance, while the weekly trend ensures alignment with higher timeframe momentum. Volume confirmation reduces false breakouts. Designed for low trade frequency (10-25/year) with clear entry/exit rules.
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
    
    # Calculate Camarilla pivot levels for the day
    # Based on previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # First value will be NaN due to roll, but we start from index 1 anyway
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r1 = close + (range_val * 1.1 / 12)  # Close-based as per some variants
    s1 = close - (range_val * 1.1 / 12)
    # Alternative: standard formula uses pivot
    # r1 = pivot + (range_val * 1.1 / 12)
    # s1 = pivot - (range_val * 1.1 / 12)
    # We'll use close-based as it's more responsive to current price action
    
    # Weekly trend filter: EMA 20 on weekly
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index 1 to avoid NaN from roll
    for i in range(1, n):
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and weekly uptrend
            if close[i] > r1[i] and volume_confirm[i]:
                # Additional filter: only take long if price above weekly EMA20 (uptrend filter)
                if close[i] > ema_20_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S1 with volume confirmation and weekly downtrend
            elif close[i] < s1[i] and volume_confirm[i]:
                # Additional filter: only take short if price below weekly EMA20 (downtrend filter)
                if close[i] < ema_20_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 (reversal signal) or weekly trend turns down
            if close[i] < s1[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 (reversal signal) or weekly trend turns up
            if close[i] > r1[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals