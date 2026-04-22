#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams %R with 1-day trend filter and volume confirmation.
Long when Williams %R oversold (< -80) and price above 1-day EMA34 and volume above average.
Short when Williams %R overbought (> -20) and price below 1-day EMA34 and volume above average.
Exit when Williams %R crosses center (-50) or volume drops below average.
Williams %R identifies overextended moves; EMA34 trend filter ensures directional bias; volume confirms participation.
Works in both bull and bear markets by capturing mean reversion within the trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for trend and volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1-day average volume (50-period) for confirmation
    avg_vol_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Williams %R (14-period) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold, price above EMA34, volume above average
            if williams_r[i] < -80 and close[i] > ema_34_1d_aligned[i] and volume[i] > avg_vol_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought, price below EMA34, volume above average
            elif williams_r[i] > -20 and close[i] < ema_34_1d_aligned[i] and volume[i] > avg_vol_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 (bullish momentum fading) OR volume drops
                if williams_r[i] > -50 or volume[i] <= avg_vol_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 (bearish momentum fading) OR volume drops
                if williams_r[i] < -50 or volume[i] <= avg_vol_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0