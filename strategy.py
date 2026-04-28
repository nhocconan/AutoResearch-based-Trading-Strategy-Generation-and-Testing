#!/usr/bin/env python3
"""
4h_Three_Sigma_Exit_12hTrend_VolumeFilter
Hypothesis: Exit positions when price moves 3 standard deviations away from 20-period mean (mean reversion in ranging markets), 
enter only when aligned with 12h EMA50 trend and volume confirmation. Works in both bull/bear by following higher timeframe trend.
Targets ~30 trades/year to minimize fee drag.
"""

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period statistics for mean reversion exit
    close_series = pd.Series(close)
    mean_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(mean_20[i]) or np.isnan(std_20[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Trend direction from 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: >1.8x 20-period MA
        vol_confirm = volume[i] > (1.8 * vol_ma_20[i])
        
        # Mean reversion bands
        upper_band = mean_20[i] + (3.0 * std_20[i])
        lower_band = mean_20[i] - (3.0 * std_20[i])
        
        # Entry conditions: trend-aligned with volume confirmation
        long_entry = trend_up and vol_confirm and (close[i] < lower_band)
        short_entry = trend_down and vol_confirm and (close[i] > upper_band)
        
        # Exit conditions: price reverts to mean (3-sigma band touch/reversal)
        long_exit = (close[i] >= mean_20[i]) or (not trend_up)
        short_exit = (close[i] <= mean_20[i]) or (not trend_down)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Three_Sigma_Exit_12hTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0