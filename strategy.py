#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 12-hour trend filter
# Long when price breaks above 20-period high AND 12h close above 12h EMA(20)
# Short when price breaks below 20-period low AND 12h close below 12h EMA(20)
# Uses 12h trend filter to avoid counter-trend trades. Discrete sizing (0.25) to minimize churn.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_donchian20_12h_ema_trend_v1"
timeframe = "4h"
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
    
    # 12-hour trend filter: EMA(20) on 12h close
    df_12h = get_htf_data(prices, '12h')
    twelve_hour_close = df_12h['close'].values
    
    # Calculate 20-period EMA on 12h close
    twelve_hour_close_series = pd.Series(twelve_hour_close)
    twelve_hour_ema = twelve_hour_close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align 12h EMA to 4h timeframe
    twelve_hour_ema_aligned = align_htf_to_ltf(prices, df_12h, twelve_hour_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if 12h EMA data not available
        if np.isnan(twelve_hour_ema_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or trend change
        if position == 1:  # long position
            # Exit: price breaks below 20-period low or 12h trend turns bearish
            if (close[i] <= donchian_low[i] or 
                twelve_hour_close[i] < twelve_hour_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 20-period high or 12h trend turns bullish
            if (close[i] >= donchian_high[i] or 
                twelve_hour_close[i] > twelve_hour_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with 12h trend filter
            # Long: price breaks above 20-period high AND 12h close above 12h EMA
            if (close[i] > donchian_high[i] and 
                twelve_hour_close[i] > twelve_hour_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low AND 12h close below 12h EMA
            elif (close[i] < donchian_low[i] and 
                  twelve_hour_close[i] < twelve_hour_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals