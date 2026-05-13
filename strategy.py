#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume
Hypothesis: Daily EMA34 sets trend direction, Camarilla R1/S1 levels provide entry points for breakouts with volume confirmation. Designed for low trade frequency (20-40/year) to work in both bull and bear markets by combining trend-following with breakout logic.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    typical = (high + low + close) / 3.0
    range_val = high - low
    r1 = close + range_val * 1.1 / 12
    r2 = close + range_val * 1.1 / 6
    r3 = close + range_val * 1.1 / 4
    r4 = close + range_val * 1.1 / 2
    s1 = close - range_val * 1.1 / 12
    s2 = close - range_val * 1.1 / 6
    s3 = close - range_val * 1.1 / 4
    s4 = close - range_val * 1.1 / 2
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and Camarilla calculation
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA34 for trend direction
    daily_close_series = pd.Series(df_daily['close'])
    ema34_daily = daily_close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily Camarilla levels
    r1_d, r2_d, r3_d, r4_d, s1_d, s2_d, s3_d, s4_d = calculate_camarilla(
        df_daily['high'].values, df_daily['low'].values, df_daily['close'].values
    )
    
    # Align daily indicators to 4h timeframe
    ema34_4h = align_htf_to_ltf(prices, df_daily, ema34_daily)
    r1_4h = align_htf_to_ltf(prices, df_daily, r1_d)
    r2_4h = align_htf_to_ltf(prices, df_daily, r2_d)
    r3_4h = align_htf_to_ltf(prices, df_daily, r3_d)
    r4_4h = align_htf_to_ltf(prices, df_daily, r4_d)
    s1_4h = align_htf_to_ltf(prices, df_daily, s1_d)
    s2_4h = align_htf_to_ltf(prices, df_daily, s2_d)
    s3_4h = align_htf_to_ltf(prices, df_daily, s3_d)
    s4_4h = align_htf_to_ltf(prices, df_daily, s4_d)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        if position == 0:
            # LONG: Price above EMA34 (uptrend) and breaks above R1 with volume
            if close[i] > ema34_4h[i] and close[i] > r1_4h[i] and close[i-1] <= r1_4h[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below EMA34 (downtrend) and breaks below S1 with volume
            elif close[i] < ema34_4h[i] and close[i] < s1_4h[i] and close[i-1] >= s1_4h[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R2 (take profit) or breaks below EMA34 (trend change)
            if close[i] >= r2_4h[i] or close[i] < ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S2 (take profit) or breaks above EMA34 (trend change)
            if close[i] <= s2_4h[i] or close[i] > ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals