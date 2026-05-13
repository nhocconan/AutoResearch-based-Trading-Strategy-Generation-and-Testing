#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeFilter
Hypothesis: Camarilla pivot levels on 12h provide strong support/resistance. A breakout above R3 or below S3 with 1w trend alignment (close > EMA50) and volume confirmation signals trend continuation. Uses 30% position size to balance risk/return and limit trade frequency (~20-30/year) to minimize fee drag in 12-hour bars.
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels for 12h: using previous bar's range
    # Camarilla: Close + (High-Low) * multiplier
    # R3 = Close + (High-Low) * 1.1/2
    # S3 = Close - (High-Low) * 1.1/2
    # We need previous bar's high/low to calculate current levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar uses current
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # 1w trend filter: EMA(50) on close
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous bar
        if position == 0:
            # LONG: Close breaks above R3, 1w uptrend, volume confirmation
            if close[i] > R3[i] and close[i] > ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: Close breaks below S3, 1w downtrend, volume confirmation
            elif close[i] < S3[i] and close[i] < ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below S3 or volume drops
            if close[i] < S3[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Close crosses above R3 or volume drops
            if close[i] > R3[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals