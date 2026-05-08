#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Uses 12h EMA50 for trend direction (aligned to 6h), Donchian breakout for entry,
# and volume > 1.5x average for confirmation. Short when price breaks below lower band.
# Trend filter ensures we only trade in the direction of higher timeframe momentum.
# Works in bull (follow breakouts up) and bear (follow breakouts down).
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "6h_DonchianBreakout_12hTrend_Volume"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = ema_50_12h > np.roll(ema_50_12h, 1)
    trend_12h_up = np.where(np.isnan(trend_12h_up), False, trend_12h_up)
    
    # Align 12h trend to 6h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    upper_channel = rolling_max(high, 20)
    lower_channel = rolling_min(low, 20)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_12h_up_aligned[i]) or np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper channel, 12h trend up, volume spike
            if (close[i] > upper_channel[i] and trend_12h_up_aligned[i] and vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel, 12h trend down, volume spike
            elif (close[i] < lower_channel[i] and not trend_12h_up_aligned[i] and vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below lower channel or trend breaks
            if (close[i] < lower_channel[i] or not trend_12h_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper channel or trend breaks
            if (close[i] > upper_channel[i] or trend_12h_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals