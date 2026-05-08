#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ehlers Fisher Transform with 1w trend filter and volume confirmation
# Fisher Transform identifies turning points in price. In bull markets, long when Fisher > -1.5;
# in bear markets, short when Fisher < +1.5. Weekly trend filter ensures we only trade
# with the higher timeframe trend. Volume confirms institutional participation.
# Targets 15-25 trades per year (~60-100 total over 4 years) to minimize fee drag.

name = "6h_Fisher_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Fisher Transform on close prices (9-period)
    price = (high + low) / 2
    max_hl = np.maximum.accumulate(high)
    min_hl = np.minimum.accumulate(low)
    price_range = max_hl - min_hl
    price_range = np.where(price_range == 0, 1, price_range)  # avoid division by zero
    
    value1 = 0.33 * 2 * ((price - min_hl) / price_range - 0.5)
    value1 = np.where(value1 >  0.99,  0.999, value1)
    value1 = np.where(value1 < -0.99, -0.999, value1)
    
    # Smooth value1
    value2 = np.zeros_like(value1)
    value2[0] = value1[0]
    for i in range(1, len(value1)):
        value2[i] = 0.5 * value1[i] + 0.5 * value2[i-1]
    
    # Fisher Transform
    fish = np.zeros_like(value2)
    fish[0] = 0
    for i in range(1, len(value2)):
        fish[i] = 0.5 * np.log((1 + value2[i]) / (1 - value2[i]) + 1e-10) + 0.5 * fish[i-1]
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1w = close_1w > ema50_1w
    trend_down_1w = close_1w < ema50_1w
    
    # Align weekly trend to 6h
    trend_up_6h = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_6h = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma.values * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient data for Fisher and MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(fish[i]) or np.isnan(trend_up_6h[i]) or np.isnan(trend_down_6h[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Fisher crosses above -1.5, weekly uptrend, volume spike
            if fish[i] > -1.5 and fish[i-1] <= -1.5 and trend_up_6h[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Fisher crosses below +1.5, weekly downtrend, volume spike
            elif fish[i] < 1.5 and fish[i-1] >= 1.5 and trend_down_6h[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Fisher crosses below -1.5 or trend changes
            if fish[i] < -1.5 or not trend_up_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Fisher crosses above +1.5 or trend changes
            if fish[i] > 1.5 or not trend_down_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals