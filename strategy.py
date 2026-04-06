#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with daily trend filter
# Long when price breaks above 12-hour high of last 20 periods AND daily close above 20-day EMA
# Short when price breaks below 12-hour low of last 20 periods AND daily close below 20-day EMA
# Uses daily trend filter to avoid counter-trend trades. Designed for low trade frequency.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range.

name = "12h_donchian20_1d_ema_trend_v1"
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
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Daily trend filter: EMA(20) on daily close
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate 20-period EMA on daily close
    daily_close_series = pd.Series(daily_close)
    daily_ema = daily_close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align daily EMA to 12h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily EMA data not available
        if np.isnan(daily_ema_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or stop loss (3*ATR)
        if position == 1:  # long position
            # Exit: price breaks below 12-hour low or daily trend turns bearish
            if (close[i] <= donchian_low[i] or 
                close[i] < daily_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 12-hour high or daily trend turns bullish
            if (close[i] >= donchian_high[i] or 
                close[i] > daily_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with daily trend filter
            # Long: price breaks above 12-hour high AND close above daily EMA
            if (close[i] > donchian_high[i] and 
                close[i] > daily_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12-hour low AND close below daily EMA
            elif (close[i] < donchian_low[i] and 
                  close[i] < daily_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals