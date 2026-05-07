#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels from previous 1d (OHLC)
    # For each day, calculate levels based on previous day's OHLC
    # We'll compute for each bar using the previous day's values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values  # previous day close
    
    # Calculate Camarilla levels for each day
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    # We need previous day's values to calculate today's levels
    prev_high = high_1d[:-1]  # shift to get previous day's high
    prev_low = low_1d[:-1]    # shift to get previous day's low
    prev_close = close_1d[:-1] # shift to get previous day's close
    
    # Calculate levels for each day (using previous day's data)
    # R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    # S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    hl_range = prev_high - prev_low
    r3_levels = prev_close + hl_range * 1.1 / 2
    s3_levels = prev_close - hl_range * 1.1 / 2
    
    # Now we need to align these levels to the 4h timeframe
    # Create arrays with same length as df_1d, where each day's levels apply to that day
    r3_per_day = np.full(len(df_1d), np.nan)
    s3_per_day = np.full(len(df_1d), np.nan)
    
    # Skip first day (no previous day), so start from index 1
    r3_per_day[1:] = r3_levels
    s3_per_day[1:] = s3_levels
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_per_day)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_per_day)
    
    # Volume spike detection (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > R3, above EMA34, volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price < S3, below EMA34, volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price < S3 or below EMA34
            if (close[i] < s3_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price > R3 or above EMA34
            if (close[i] > r3_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Camarilla levels identify key support/resistance from previous day's price action.
# Breakout above R3 with volume suggests bullish momentum; breakdown below S3 suggests bearish.
# EMA34 filter ensures we only trade in the direction of the daily trend.
# Volume confirmation ensures institutional participation.
# Works in bull markets (buy breakouts above R3 in uptrend) and bear markets (sell breakdowns below S3 in downtrend).
# Position size 0.25 balances risk and keeps trade frequency manageable (~20-40 trades/year).