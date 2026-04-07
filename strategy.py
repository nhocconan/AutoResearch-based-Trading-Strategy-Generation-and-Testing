#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: On 1d timeframe, buy when price breaks above 20-day Donchian upper band with volume confirmation and weekly uptrend (price > weekly EMA50), sell/short when price breaks below 20-day lower band with volume confirmation and weekly downtrend (price < weekly EMA50). This captures medium-term trends while avoiding false breakouts in ranging markets. Weekly trend filter ensures alignment with higher timeframe momentum, reducing whipsaws. Volume confirmation ensures breakouts are supported by participation. Designed for low trade frequency (<25/year) to minimize fee impact in both bull and bear markets.
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
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    weekly_ema50 = df_weekly['close'].ewm(span=50, adjust=False).mean()
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50.values)
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-day low or weekly trend turns bearish
            if close[i] < low_roll[i] or close[i] < weekly_ema50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above 20-day high or weekly trend turns bullish
            if close[i] > high_roll[i] or close[i] > weekly_ema50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-day high with volume in weekly uptrend
            if (close[i] > high_roll[i] and 
                vol_confirm and 
                close[i] > weekly_ema50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-day low with volume in weekly downtrend
            elif (close[i] < low_roll[i] and 
                  vol_confirm and 
                  close[i] < weekly_ema50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals