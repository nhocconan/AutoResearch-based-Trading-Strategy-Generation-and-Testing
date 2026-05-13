#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance. Breakout of these levels with volume confirmation and aligned 1d EMA34 trend provides high-probability entries. This strategy targets fewer than 50 trades/year to minimize fee drag while capturing strong intraday moves in both bull and bear markets.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume"
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 1d bar
    # R3 = close + (high - low) * 1.1 / 4
    # S3 = close - (high - low) * 1.1 / 4
    hl_range = df_1d['high'] - df_1d['low']
    camarilla_r3 = df_1d['close'] + hl_range * 1.1 / 4
    camarilla_s3 = df_1d['close'] - hl_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # 1d trend filter: EMA(34) on close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.8x 24-period average (1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        if position == 0:
            # LONG: Close breaks above R3, volume confirmation, price above 1d EMA34 (uptrend)
            if close[i] > r3_aligned[i] and volume_filter[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S3, volume confirmation, price below 1d EMA34 (downtrend)
            elif close[i] < s3_aligned[i] and volume_filter[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close breaks below S3 (reversal) or volume drops
            if close[i] < s3_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close breaks above R3 (reversal) or volume drops
            if close[i] > r3_aligned[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals