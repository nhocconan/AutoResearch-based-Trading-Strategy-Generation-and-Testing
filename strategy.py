#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
Breakouts from 20-period high/low capture momentum, daily trend filter avoids counter-trend trades,
volume spikes confirm institutional interest. Works in bull markets (breakouts up) and bear markets
(breakouts down) by following the daily trend. Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Determine daily trend: price above/below EMA34
    bullish_trend = close_1d > ema_34_1d
    bearish_trend = close_1d < ema_34_1d
    
    # Align daily EMA and trend to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # Calculate 12h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bullish_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, bullish daily trend, volume spike
            if (close[i] > high_max_20[i] and         # Breakout above 20-period high
                close[i] > ema_34_aligned[i] and      # Price above daily EMA34 (redundant but safe)
                bullish_aligned[i] > 0.5 and          # Bullish daily trend
                volume[i] > 2.0 * vol_avg_20[i]):     # Volume spike (2x average)
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, bearish daily trend, volume spike
            elif (close[i] < low_min_20[i] and        # Breakout below 20-period low
                  close[i] < ema_34_aligned[i] and    # Price below daily EMA34 (redundant but safe)
                  bearish_aligned[i] > 0.5 and        # Bearish daily trend
                  volume[i] > 2.0 * vol_avg_20[i]):   # Volume spike (2x average)
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Donchian midpoint (mean reversion within channel)
            donchian_mid = (high_max_20[i] + low_min_20[i]) / 2
            
            if position == 1:
                # Exit long: price drops below midpoint
                if close[i] < donchian_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price rises above midpoint
                if close[i] > donchian_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0