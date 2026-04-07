#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_trend_volume_v1
Hypothesis: On 12h timeframe, buy when price breaks above 20-period Donchian upper band with 1d EMA50 > EMA200 and volume > 1.5x average; sell when price breaks below 20-period Donchian lower band with 1d EMA50 < EMA200 and volume > 1.5x average. Exit when price crosses the opposite Donchian band or EMA trend reverses. Uses Donchian breakouts for trend continuation, EMA trend filter for direction, and volume for confirmation. Works in bull/bear via EMA trend filter. Targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA trend filter (50 and 200)
    close_1d = pd.Series(close)
    ema50_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = close_1d.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (24-period average = 12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema50_1d[i]) or np.isnan(ema200_1d[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price crosses below Donchian lower band
            if close[i] < donchian_low[i]:
                exit_long = True
            # Exit if EMA50 crosses below EMA200 (trend reversal)
            elif ema50_1d[i] < ema200_1d[i] and ema50_1d[i-1] >= ema200_1d[i-1]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price crosses above Donchian upper band
            if close[i] > donchian_high[i]:
                exit_short = True
            # Exit if EMA50 crosses above EMA200 (trend reversal)
            elif ema50_1d[i] > ema200_1d[i] and ema50_1d[i-1] <= ema200_1d[i-1]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper with EMA50 > EMA200 and volume confirmation
            long_entry = False
            if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] and
                ema50_1d[i] > ema200_1d[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price breaks below Donchian lower with EMA50 < EMA200 and volume confirmation
            short_entry = False
            if (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] and
                ema50_1d[i] < ema200_1d[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals