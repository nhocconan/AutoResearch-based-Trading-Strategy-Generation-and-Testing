#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with daily trend filter and volume confirmation.
Long when price breaks above 20-bar Donchian high AND daily EMA34 trend is up AND volume > 1.5x average.
Short when price breaks below 20-bar Donchian low AND daily EMA34 trend is down AND volume > 1.5x average.
Exit when price returns to the Donchian midpoint or volume drops below average.
Designed for low-frequency, high-quality trades (~15-30/year) to capture strong trends while avoiding chop.
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
    
    # Load daily data for EMA trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12-hour Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + daily uptrend + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_34_aligned[i] and 
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + daily downtrend + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to Donchian midpoint OR volume drops below average
                if close[i] <= donchian_mid[i] or volume[i] < vol_avg[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to Donchian midpoint OR volume drops below average
                if close[i] >= donchian_mid[i] or volume[i] < vol_avg[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0