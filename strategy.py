#!/usr/bin/env python3
"""
1h_Camarilla_Pivots_Breakout_Trend_Filter
Hypothesis: Camarilla pivot levels (R3/S3) on 4h act as strong support/resistance. Breakouts above R3 or below S3 with 1d trend filter (close > EMA50) and volume confirmation signal continuation. Uses 1h only for entry timing to reduce false breaks. Target 15-30 trades/year to minimize fee drag. Works in bull (breakouts continue) and bear (breakdowns continue) via trend filter.
"""

name = "1h_Camarilla_Pivots_Breakout_Trend_Filter"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla pivots (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels for each 4h bar
    # R4 = close + (high - low) * 1.5
    # R3 = close + (high - low) * 1.25/2
    # S3 = close - (high - low) * 1.25/2
    # S4 = close - (high - low) * 1.5
    hl_range = df_4h['high'].values - df_4h['low'].values
    r3_4h = df_4h['close'].values + hl_range * 0.625  # 1.25/2 = 0.625
    s3_4h = df_4h['close'].values - hl_range * 0.625
    
    # Align Camarilla levels to 1h (wait for 4h bar close)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # 1d trend filter: EMA(50) on close
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        if position == 0:
            # LONG: Close crosses above R3 with volume and uptrend
            if (close[i] > r3_4h_aligned[i] and close[i-1] <= r3_4h_aligned[i-1] and
                volume_filter[i] and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Close crosses below S3 with volume and downtrend
            elif (close[i] < s3_4h_aligned[i] and close[i-1] >= s3_4h_aligned[i-1] and
                  volume_filter[i] and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below S3 (mean reversion) or trend fails
            if (close[i] < s3_4h_aligned[i] and close[i-1] >= s3_4h_aligned[i-1]) or \
               close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close crosses above R3 (mean reversion) or trend fails
            if (close[i] > r3_4h_aligned[i] and close[i-1] <= r3_4h_aligned[i-1]) or \
               close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals