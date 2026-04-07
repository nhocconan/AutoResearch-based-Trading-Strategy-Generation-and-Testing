#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: On daily timeframe, trade Donchian(20) breakouts with 1-week EMA trend filter and volume confirmation.
Breakouts above upper band (long) or below lower band (short) only when aligned with weekly trend and confirmed by volume.
In uptrend (price > weekly EMA50): long breakouts, short breakdowns.
In downtrend (price < weekly EMA50): short breakdowns, long breakouts (counter-trend fade).
Volume ensures genuine breakouts. Target: 20-50 total trades over 4 years (5-12/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = df_1w['close'].ewm(span=50, adjust=False).mean()
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w.values)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band
            if close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band
            if close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above upper band with volume in uptrend
            if (close[i] > high_20[i] and
                vol_confirm and 
                close[i] > ema_50_1w_aligned[i]):  # uptrend filter
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below lower band with volume in downtrend
            elif (close[i] < low_20[i] and
                  vol_confirm and 
                  close[i] < ema_50_1w_aligned[i]):  # downtrend filter
                position = -1
                signals[i] = -0.25
            # Counter-trend fade in ranging markets: fade extreme touches
            elif (vol_confirm and
                  abs(close[i] - high_20[i]) < 0.001 * high_20[i] and  # touching upper band
                  close[i] < ema_50_1w_aligned[i]):  # in downtrend, fade upper touch
                position = -1
                signals[i] = -0.20
            elif (vol_confirm and
                  abs(close[i] - low_20[i]) < 0.001 * low_20[i] and  # touching lower band
                  close[i] > ema_50_1w_aligned[i]):  # in uptrend, fade lower touch
                position = 1
                signals[i] = 0.20
    
    return signals