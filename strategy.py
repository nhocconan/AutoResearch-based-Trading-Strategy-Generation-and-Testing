#!/usr/bin/env python3
# 1d_1w_Donchian_Breakout_TrendFilter_Volume
# Hypothesis: Weekly trend bias with daily Donchian breakouts and volume confirmation.
# In bull markets: buy breakouts above 20-day high when weekly trend is up.
# In bear markets: sell breakdowns below 20-day low when weekly trend is down.
# Volume surge filter reduces false breakouts. Designed for low trade frequency.

name = "1d_1w_Donchian_Breakout_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend: EMA10 slope
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_slope_10_1w = np.diff(ema_10_1w, prepend=ema_10_1w[0])
    ema_slope_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_10_1w)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_slope_10_1w_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        bullish_trend = ema_slope_10_1w_aligned[i] > 0
        bearish_trend = ema_slope_10_1w_aligned[i] < 0
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above 20-day high in bullish weekly trend with volume
            if high[i] > high_20[i-1] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below 20-day low in bearish weekly trend with volume
            elif low[i] < low_20[i-1] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: close below 10-day EMA or weekly trend turns bearish
                ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
                if close[i] < ema_10[i] or ema_slope_10_1w_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: close above 10-day EMA or weekly trend turns bullish
                ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
                if close[i] > ema_10[i] or ema_slope_10_1w_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals