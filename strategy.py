#!/usr/bin/env python3
# 6h_Premium_Discount_Zone_With_Volume_Trend
# Hypothesis: Buy near discount zone (below 50% of weekly range) in uptrend (price > 12-period EMA) and sell near premium zone (above 50% of weekly range) in downtrend (price < 12-period EMA) on 6b timeframe, with volume confirmation (volume > 1.5x 20-period average). Uses weekly range for structure and 12-period EMA for trend filter to work in both bull and bear markets. Volume confirmation reduces whipsaw. Designed for 6h timeframe to balance trade frequency and capture mean reversion within weekly cycles.

name = "6h_Premium_Discount_Zone_With_Volume_Trend"
timeframe = "6h"
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
    
    # Get weekly data for range calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly range and 50% level (based on completed weekly bars)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_range = weekly_high - weekly_low
    weekly_mid = weekly_low + 0.5 * weekly_range
    
    # Align weekly levels to 6h timeframe (wait for weekly close)
    weekly_mid_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    # Get 12-period EMA for trend filter on 6h timeframe
    close_series = pd.Series(close)
    ema_12 = close_series.ewm(span=12, adjust=False, min_periods=12).values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price below weekly midpoint (discount zone) with volume confirmation in uptrend
            if weekly_mid_aligned[i] > 0 and not np.isnan(weekly_mid_aligned[i]) and \
               close[i] < weekly_mid_aligned[i] and volume_confirmed[i] and close[i] > ema_12[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price above weekly midpoint (premium zone) with volume confirmation in downtrend
            elif weekly_mid_aligned[i] > 0 and not np.isnan(weekly_mid_aligned[i]) and \
                 close[i] > weekly_mid_aligned[i] and volume_confirmed[i] and close[i] < ema_12[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above weekly midpoint or trend weakens
            if weekly_mid_aligned[i] > 0 and not np.isnan(weekly_mid_aligned[i]) and \
               close[i] > weekly_mid_aligned[i] or close[i] < ema_12[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below weekly midpoint or trend weakens
            if weekly_mid_aligned[i] > 0 and not np.isnan(weekly_mid_aligned[i]) and \
               close[i] < weekly_mid_aligned[i] or close[i] > ema_12[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals