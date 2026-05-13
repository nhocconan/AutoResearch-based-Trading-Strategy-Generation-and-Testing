#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot breakouts on 12h with 1-week trend filter and volume confirmation capture institutional moves. 
Designed for low trade frequency (12-37/year) by combining price structure (Camarilla levels), 
trend alignment (1-week EMA), and volume filters. Works in both bull and bear markets by 
using trend-appropriate breakouts (long in uptrend, short in downtrend).
"""

name = "12h_Camarilla_Pivot_Breakout_1wTrend_Volume"
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
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # Daily pivot points from previous period
    prev_typical = np.roll(typical_price, 1)
    prev_typical[0] = typical_price[0]  # first value
    
    # Calculate daily range
    daily_range = high - low
    prev_range = np.roll(daily_range, 1)
    prev_range[0] = daily_range[0]
    
    # Camarilla levels (based on previous day)
    R4 = prev_typical + (prev_range * 1.1 / 2)
    R3 = prev_typical + (prev_range * 1.1 / 4)
    R2 = prev_typical + (prev_range * 1.1 / 6)
    R1 = prev_typical + (prev_range * 1.1 / 12)
    S1 = prev_typical - (prev_range * 1.1 / 12)
    S2 = prev_typical - (prev_range * 1.1 / 6)
    S3 = prev_typical - (prev_range * 1.1 / 4)
    S4 = prev_typical - (prev_range * 1.1 / 2)
    
    # Volume confirmation: > 1.5x 24-period average (2 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Get 1-week trend filter (EMA 20)
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        if position == 0:
            # LONG: Price breaks above R4 with volume confirmation and 1-week uptrend
            if close[i] > R4[i] and volume_confirm[i]:
                # Additional filter: only take long if price above 1-week EMA20 (uptrend filter)
                if close[i] > ema_20_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S4 with volume confirmation and 1-week downtrend
            elif close[i] < S4[i] and volume_confirm[i]:
                # Additional filter: only take short if price below 1-week EMA20 (downtrend filter)
                if close[i] < ema_20_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below R1 (mean reversion) or trend change
            if close[i] < R1[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above S1 (mean reversion) or trend change
            if close[i] > S1[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals