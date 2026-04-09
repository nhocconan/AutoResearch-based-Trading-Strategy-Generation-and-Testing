#!/usr/bin/env python3
# 12h_donchian_breakout_1w_trend_volume_v1
# Hypothesis: 12h Donchian(20) breakout with 1w EMA trend filter and 12h volume confirmation.
# Weekly EMA(50) determines primary trend direction (only long above, short below).
# 12h Donchian(20) breakout provides entry in trend direction.
# 12h volume > 1.5x 20-period average confirms institutional participation.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-30 trades/year.
# Uses weekly HTF and 12h data called ONCE before loop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h data for Donchian channels and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 12h indicators to primary timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    volume_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h volume (need to align raw volume)
        # Since we don't have a function to align raw volume directly,
        # we use the close price index to get the corresponding 12h bar
        # For 12h timeframe, each 12h bar = 48 primary bars (if primary is 15m)
        # But we don't know primary timeframe here, so we use a different approach:
        # We'll use the volume from the prices dataframe directly and compare
        # to the aligned volume MA (which represents the 20-period MA of 12h volume)
        # This works because volume_ma_12h_aligned[i] is the 20-period MA of 12h volume
        # aligned to the current bar, so we compare current volume to it
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low OR weekly EMA turns bearish (price < EMA)
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR weekly EMA turns bullish (price > EMA)
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation: current volume > 1.5x 20-period average of 12h volume
            volume_confirmed = volume[i] > 1.5 * volume_ma_12h_aligned[i]
            
            if volume_confirmed:
                # Long entry: price breaks above Donchian high AND above weekly EMA (uptrend)
                if close[i] > donchian_high_aligned[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low AND below weekly EMA (downtrend)
                elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals