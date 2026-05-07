#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for weekly trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d (OHLC)
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values  # previous day close
    
    # Calculate levels for each day (using previous day's data)
    # Skip first day (no previous day)
    prev_high = high_1d[:-1]  # previous day's high
    prev_low = low_1d[:-1]    # previous day's low
    prev_close = close_1d[:-1] # previous day's close
    
    hl_range = prev_high - prev_low
    r3_levels = prev_close + hl_range * 1.1 / 2
    s3_levels = prev_close - hl_range * 1.1 / 2
    
    # Create arrays with same length as df_1d, where each day's levels apply to that day
    r3_per_day = np.full(len(df_1d), np.nan)
    s3_per_day = np.full(len(df_1d), np.nan)
    
    # Skip first day (no previous day), so start from index 1
    r3_per_day[1:] = r3_levels
    s3_per_day[1:] = s3_levels
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_per_day)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_per_day)
    
    # Volume spike detection (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > R3, above weekly EMA50, volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price < S3, below weekly EMA50, volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price < S3 or below weekly EMA50
            if (close[i] < s3_aligned[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price > R3 or above weekly EMA50
            if (close[i] > r3_aligned[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with weekly EMA50 trend filter and volume confirmation.
# Weekly EMA50 ensures we trade only in the direction of the long-term trend.
# Camarilla levels identify key support/resistance from previous day's price action.
# Breakout above R3 with volume suggests bullish momentum; breakdown below S3 suggests bearish.
# Volume confirmation ensures institutional participation.
# Works in bull markets (buy breakouts above R3 in uptrend) and bear markets (sell breakdowns below S3 in downtrend).
# Position size 0.25 balances risk and keeps trade frequency low (~15-30 trades/year).