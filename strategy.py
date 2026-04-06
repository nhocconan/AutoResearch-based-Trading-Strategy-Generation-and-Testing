#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian breakout with 1-day volume confirmation and 1-week trend filter
# Long when price breaks above 6h 20-period Donchian high, 1d volume > 1.5x 20-period average, and weekly close > 20-week EMA
# Short when price breaks below 6h 20-period Donchian low, 1d volume > 1.5x 20-period average, and weekly close < 20-week EMA
# Uses volume to confirm breakout strength and weekly trend to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) for optimal statistical significance.

name = "6h_donchian20_1d_vol_1w_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # 1-day volume average (20-period)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean()
    volume_ratio = volume / volume_ma  # Current volume / 20-period average
    
    # 1-week trend filter: EMA(20) on weekly close
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly close
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema = weekly_close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align weekly EMA to 6h timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly EMA or volume MA data not available
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or trend change
        if position == 1:  # long position
            # Exit: price breaks below 6h 20-period low or weekly trend turns bearish
            if (close[i] <= donchian_low[i] or 
                weekly_close[i] < weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 6h 20-period high or weekly trend turns bullish
            if (close[i] >= donchian_high[i] or 
                weekly_close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and weekly trend filter
            # Long: price breaks above 6h 20-period high AND volume > 1.5x average AND weekly close above weekly EMA
            if (close[i] > donchian_high[i] and 
                volume_ratio[i] > 1.5 and 
                weekly_close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h 20-period low AND volume > 1.5x average AND weekly close below weekly EMA
            elif (close[i] < donchian_low[i] and 
                  volume_ratio[i] > 1.5 and 
                  weekly_close[i] < weekly_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals