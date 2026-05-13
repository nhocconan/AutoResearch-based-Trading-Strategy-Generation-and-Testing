#!/usr/bin/env python3
"""
4h_Support_Resistance_Bounce_Turn_12hTrend_Filter
Hypothesis: Price bounces off multi-timeframe support/resistance levels (prior swing highs/lows) with trend alignment from 12h EMA and volume confirmation. Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend). Uses swing points for structure, EMA for trend filter, volume for confirmation. Targets 20-40 trades/year to minimize fee drag.
"""

name = "4h_Support_Resistance_Bounce_Turn_12hTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get swing points from 1d timeframe for S/R levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate swing highs and lows (3-bar lookback/forward)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    swing_high = np.full_like(high_1d, np.nan)
    swing_low = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(high_1d) - 1):
        if high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1]:
            swing_high[i] = high_1d[i]
        if low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1]:
            swing_low[i] = low_1d[i]
    
    # Forward fill swing levels to create support/resistance zones
    swing_high_ff = pd.Series(swing_high).ffill().values
    swing_low_ff = pd.Series(swing_low).ffill().values
    
    # Align S/R levels to 4h timeframe
    resistance_aligned = align_htf_to_ltf(prices, df_1d, swing_high_ff)
    support_aligned = align_htf_to_ltf(prices, df_1d, swing_low_ff)
    
    # Get 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    ema_30_12h = pd.Series(df_12h['close']).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_30_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit
    
    for i in range(50, n):
        bars_since_exit += 1
        
        if position == 0:
            # LONG: Price near support with bullish alignment
            if (support_aligned[i] > 0 and not np.isnan(support_aligned[i]) and
                low[i] <= support_aligned[i] * 1.005 and  # within 0.5% of support
                close[i] > support_aligned[i] and        # closing above support
                ema_30_12h_aligned[i] > 0 and not np.isnan(ema_30_12h_aligned[i]) and
                close[i] > ema_30_12h_aligned[i] and     # above 12h EMA (uptrend)
                volume_confirmed[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # SHORT: Price near resistance with bearish alignment
            elif (resistance_aligned[i] > 0 and not np.isnan(resistance_aligned[i]) and
                  high[i] >= resistance_aligned[i] * 0.995 and  # within 0.5% of resistance
                  close[i] < resistance_aligned[i] and          # closing below resistance
                  ema_30_12h_aligned[i] > 0 and not np.isnan(ema_30_12h_aligned[i]) and
                  close[i] < ema_30_12h_aligned[i] and        # below 12h EMA (downtrend)
                  volume_confirmed[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position == 1:
            # EXIT LONG: Price breaks below support or trend changes
            if (support_aligned[i] > 0 and not np.isnan(support_aligned[i]) and
                close[i] < support_aligned[i]) or \
               (ema_30_12h_aligned[i] > 0 and not np.isnan(ema_30_12h_aligned[i]) and
                close[i] < ema_30_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # EXIT SHORT: Price breaks above resistance or trend changes
            if (resistance_aligned[i] > 0 and not np.isnan(resistance_aligned[i]) and
                close[i] > resistance_aligned[i]) or \
               (ema_30_12h_aligned[i] > 0 and not np.isnan(ema_30_12h_aligned[i]) and
                close[i] > ema_30_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals