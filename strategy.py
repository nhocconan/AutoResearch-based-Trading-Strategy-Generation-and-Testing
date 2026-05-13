#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R3/S3) from 1-day chart act as significant support/resistance. 
Breakouts above R3 or below S3 with volume confirmation and aligned 1-day trend (EMA 50) capture strong momentum moves. 
Designed for low trade frequency (15-25/year) to avoid fee drag, with trend-following logic that works in both bull and bear markets.
"""

name = "12h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume"
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
    
    # Calculate Camarilla pivot levels for 12h period
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_range = (high - low) * 1.1
    r3 = close + camarilla_range / 2
    s3 = close - camarilla_range / 2
    
    # Volume confirmation: > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Get 1-day trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if position == 0:
            # LONG: Price breaks above R3 with volume confirmation and uptrend filter
            if close[i] > r3[i] and volume_confirm[i]:
                # Additional filter: only take long if price above 1-day EMA50 (uptrend filter)
                if close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S3 with volume confirmation and downtrend filter
            elif close[i] < s3[i] and volume_confirm[i]:
                # Additional filter: only take short if price below 1-day EMA50 (downtrend filter)
                if close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 (reversion to mean) or volume drops
            if close[i] < s3[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 (reversion to mean) or volume drops
            if close[i] > r3[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals