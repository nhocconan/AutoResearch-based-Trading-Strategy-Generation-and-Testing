#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrend_Volume
Hypothesis: Camarilla R3/S3 breakout with 1-week EMA trend filter and volume confirmation.
Uses 12h timeframe for lower trade frequency (target: 15-30/year) to minimize fee drag.
Combines price structure (Camarilla pivots), trend filter (1w EMA), and volume confirmation.
Designed to work in both bull and bear markets by following the higher timeframe trend.
"""

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
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
    
    # Calculate Camarilla levels using previous day's OHLC
    # Typical price for pivot calculation
    typical_price = (high + low + close) / 3
    # Daily range
    daily_range = high - low
    
    # Camarilla levels (based on previous period)
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_width = daily_range * 1.1 / 2
    r3 = close + camarilla_width
    s3 = close - camarilla_width
    
    # Shift levels to avoid look-ahead (use previous bar's levels)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    r3[0] = 0
    s3[0] = 0
    
    # Get 1-week trend filter (EMA 34)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation and uptrend filter
            if close[i] > r3[i] and volume_confirm[i]:
                # Additional filter: only take long if price above 1-week EMA34 (uptrend)
                if close[i] > ema_34_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S3 with volume confirmation and downtrend filter
            elif close[i] < s3[i] and volume_confirm[i]:
                # Additional filter: only take short if price below 1-week EMA34 (downtrend)
                if close[i] < ema_34_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 (reversal) or volume dries up
            if close[i] < s3[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 (reversal) or volume dries up
            if close[i] > r3[i] or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals