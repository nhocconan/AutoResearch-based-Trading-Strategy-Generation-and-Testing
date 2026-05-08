#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h EMA trend filter with 4h Donchian breakout and volume confirmation.
# Uses 12h EMA for trend direction (avoids whipsaws), 4h Donchian(20) breakout for entry timing,
# and 4h volume spike (2x 20-period EMA) for momentum confirmation.
# Designed for moderate trade frequency (~30-50/year) to balance opportunity and fee drag.
# Works in bull markets by following uptrend breakouts and in bear markets by shorting downtrend breakdowns.

name = "4h_Donchian_EMA12h_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA (34-period)
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h Donchian channel (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Calculate 4h volume spike (2x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian high + 12h EMA uptrend + volume spike
            if close[i] > donchian_high[i] and ema_12h_aligned[i] < close[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 4h Donchian low + 12h EMA downtrend + volume spike
            elif close[i] < donchian_low[i] and ema_12h_aligned[i] > close[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low or 12h EMA turns down
            if close[i] < donchian_low[i] or ema_12h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high or 12h EMA turns up
            if close[i] > donchian_high[i] or ema_12h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals