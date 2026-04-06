#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and 1w trend filter.
# Long when price breaks above Donchian(20) high with daily volume > 1.5x weekly average.
# Short when price breaks below Donchian(20) low with daily volume > 1.5x weekly average.
# Weekly trend filter: only trade long when price > weekly EMA(50), short when < weekly EMA(50).
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range.
# Uses 12h timeframe for lower frequency, daily volume for confirmation, weekly EMA for trend filter.

name = "12h_donchian20_1d_vol_1w_ema_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # 1d volume average (daily volume, then average)
    df_1d = get_htf_data(prices, '1d')
    daily_volume = df_1d['volume'].values
    daily_vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    daily_vol_ma_aligned = align_htf_to_ltf(prices, df_1d, daily_vol_ma)
    
    # 1w EMA trend filter (50-period)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly trend data not available
        if np.isnan(weekly_ema_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss via signal change
        if position == 1:  # long position
            # Exit: price breaks below Donchian low or weekly trend turns bearish
            if (close[i] < donchian_low[i] or 
                close[i] < weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high or weekly trend turns bullish
            if (close[i] > donchian_high[i] or 
                close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Check for entries with volume confirmation
            # Volume condition: daily volume > 1.5x weekly average volume
            vol_condition = volume[i] > 1.5 * daily_vol_ma_aligned[i]
            
            # Long: price breaks above Donchian high with volume confirmation and bullish weekly trend
            if (close[i] > donchian_high[i] and 
                vol_condition and 
                close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume confirmation and bearish weekly trend
            elif (close[i] < donchian_low[i] and 
                  vol_condition and 
                  close[i] < weekly_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals