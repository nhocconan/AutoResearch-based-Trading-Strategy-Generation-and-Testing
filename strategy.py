#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels on 12h chart combined with 1-day EMA trend filter and volume confirmation.
Captures breakouts from key support/resistance levels with institutional relevance.
Designed for low trade frequency (15-25/year) to avoid fee drag while capturing significant moves.
Works in both bull and bear markets by following the daily trend direction.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Calculate Camarilla levels for R3 and S3 (based on previous day's range)
    # For 12h chart, we need to calculate based on previous 12h bar's range
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # handle first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Calculate range
    daily_range = prev_high - prev_low
    
    # Camarilla R3 and S3 levels
    r3 = prev_close + (daily_range * 1.1 / 4)
    s3 = prev_close - (daily_range * 1.1 / 4)
    
    # Get 1-day trend filter (EMA 34)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation and uptrend
            if close[i] > r3[i] and volume_confirm[i]:
                # Additional filter: only take long if price above 1-day EMA34 (uptrend)
                if close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S3 with volume confirmation and downtrend
            elif close[i] < s3[i] and volume_confirm[i]:
                # Additional filter: only take short if price below 1-day EMA34 (downtrend)
                if close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 (key support) or trend changes
            if close[i] < s3[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 (key resistance) or trend changes
            if close[i] > r3[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals