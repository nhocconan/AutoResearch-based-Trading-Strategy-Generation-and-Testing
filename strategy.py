#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-period (10-day) high AND weekly close above 20-week EMA AND volume > 1.5x 20-period average volume.
# Short when price breaks below 20-period low AND weekly close below 20-week EMA AND volume > 1.5x average.
# Uses weekly trend filter to avoid counter-trend trades and volume confirmation to avoid false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
# Works in bull markets via trend-following breakouts and in bear markets via short-side breakouts with trend filter.

name = "12h_donchian20_1w_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Average volume for confirmation
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=20, min_periods=20).mean()
    
    # Weekly trend filter: EMA(20) on weekly close
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly close
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema = weekly_close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align weekly EMA to 12h timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly EMA data not available
        if np.isnan(weekly_ema_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or trend change
        if position == 1:  # long position
            # Exit: price breaks below 20-period low or weekly trend turns bearish
            if (close[i] <= donchian_low[i] or 
                weekly_close[i] < weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 20-period high or weekly trend turns bullish
            if (close[i] >= donchian_high[i] or 
                weekly_close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with weekly trend filter and volume confirmation
            vol_ok = volume[i] > 1.5 * avg_volume[i] if not np.isnan(avg_volume[i]) else False
            
            # Long: price breaks above 20-period high AND weekly close above weekly EMA AND volume confirmation
            if (close[i] > donchian_high[i] and 
                weekly_close[i] > weekly_ema_aligned[i] and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low AND weekly close below weekly EMA AND volume confirmation
            elif (close[i] < donchian_low[i] and 
                  weekly_close[i] < weekly_ema_aligned[i] and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
    
    return signals