#!/usr/bin/env python3
# 6h_Fisher_Transform_Breakout_1D_Trend_Filter
# Hypothesis: The Ehlers Fisher Transform identifies turning points in price cycles.
# Long when Fisher crosses above -1.5 AND price is above 1D EMA50 (bullish trend).
# Short when Fisher crosses below +1.5 AND price is below 1D EMA50 (bearish trend).
# The 1D EMA50 acts as a trend filter to avoid counter-trend trades.
# Works in bull/bear by following the higher timeframe trend.

name = "6h_Fisher_Transform_Breakout_1D_Trend_Filter"
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

    # Get 1d data for EMA50 calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Ehlers Fisher Transform on 6h close (9-period)
    price = (high + low) / 2.0
    # Normalize price to [-1, 1] over 9 periods
    max_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    min_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    # Avoid division by zero
    price_range = max_high - min_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    value1 = 2.0 * ((price - min_low) / price_range - 0.5)
    # Smooth with 2-period EMA
    value1_smooth = pd.Series(value1).ewm(span=2, adjust=False, min_periods=2).mean().values
    # Clamp to [-0.999, 0.999] for Fisher transform
    value1_smooth = np.clip(value1_smooth, -0.999, 0.999)
    # Fisher Transform
    fish = 0.5 * np.log((1.0 + value1_smooth) / (1.0 - value1_smooth))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(9, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(fish[i]) or 
            np.isnan(fish[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Fisher crosses above -1.5 AND price above 1D EMA50 (bullish trend)
            if fish[i] > -1.5 and fish[i-1] <= -1.5 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Fisher crosses below +1.5 AND price below 1D EMA50 (bearish trend)
            elif fish[i] < 1.5 and fish[i-1] >= 1.5 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fisher crosses below +1.5 OR price below 1D EMA50
            if fish[i] < 1.5 and fish[i-1] >= 1.5 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fisher crosses above -1.5 OR price above 1D EMA50
            if fish[i] > -1.5 and fish[i-1] <= -1.5 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals