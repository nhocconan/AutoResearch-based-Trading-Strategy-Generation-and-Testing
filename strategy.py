#!/usr/bin/env python3
# 12h_Camarilla_R3S3_1dTrend_Volume_Confirm
# Hypothesis: Use 12h timeframe with Camarilla R3/S3 breakout, confirmed by 1d EMA34 trend and volume spike.
# Targets 15-25 trades/year to minimize fee drift. Works in bull (breakouts with trend) and bear (mean reversion at extremes).
# Volume filter reduces false breakouts; trend filter ensures directional bias.

name = "12h_Camarilla_R3S3_1dTrend_Volume_Confirm"
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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels: R3, S3
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 4
    s3 = close_1d - 1.1 * camarilla_range / 4
    
    # Get 1d data for trend filter (EMA34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter on 12h (24-period average = ~12 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema_34_1d_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: Price > R3, above 1d EMA34 trend, volume spike
            if close[i] > r3_12h[i] and close[i] > ema_34_1d_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Price < S3, below 1d EMA34 trend, volume spike
            elif close[i] < s3_12h[i] and close[i] < ema_34_1d_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price closes below R3 OR below trend
            if close[i] < r3_12h[i] or close[i] < ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above S3 OR above trend
            if close[i] > s3_12h[i] or close[i] > ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals