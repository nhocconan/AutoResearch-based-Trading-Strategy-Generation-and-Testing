#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 1-week EMA(200) trend filter and volume confirmation
# Uses Donchian breakouts on 6h timeframe for entry signals
# Filters by 1-week EMA(200) to only trade in the direction of the higher timeframe trend
# Requires volume > 1.5x 20-period average for confirmation
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear markets: captures breakouts in trending markets, avoids false signals in ranges

name = "6h_Donchian20_1wEMA200_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1-week EMA(200) for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 6h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation filter (>1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high AND above 1w EMA200 AND volume confirmation
            if (close[i] > donchian_high[i] and close[i] > ema_200_1w_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND below 1w EMA200 AND volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < ema_200_1w_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when price breaks below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals