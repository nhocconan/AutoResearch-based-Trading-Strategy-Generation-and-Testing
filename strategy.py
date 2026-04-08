#!/usr/bin/env python3
# 1d_donchian_breakout_1w_trend_volume_v2
# Hypothesis: Daily Donchian(20) breakouts with weekly EMA40 trend filter and volume confirmation.
# Only long when price breaks above Donchian upper band in uptrend (price > weekly EMA40),
# only short when price breaks below Donchian lower band in downtrend (price < weekly EMA40).
# Volume must be above 1.5x 20-day average to confirm breakout strength.
# Fixed position size 0.25 to limit risk. Designed for low frequency (<25 trades/year) to avoid fee drag.
# Works in bull markets via trend-following breakouts and in bear via short breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Calculate 20-day average volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-day)
    # Upper band = highest high of last 20 days
    # Lower band = lowest low of last 20 days
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_40_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band OR trend turns against us
            if (close[i] < donchian_lower[i]) or (close[i] < ema_40_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band OR trend turns against us
            if (close[i] > donchian_upper[i]) or (close[i] > ema_40_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper band with uptrend and volume confirmation
            if (close[i] > donchian_upper[i]) and (close[i] > ema_40_1w_aligned[i]) and (vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band with downtrend and volume confirmation
            elif (close[i] < donchian_lower[i]) and (close[i] < ema_40_1w_aligned[i]) and (vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals