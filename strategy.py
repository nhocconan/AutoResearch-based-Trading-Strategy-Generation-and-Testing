#!/usr/bin/env python3
name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate Camarilla levels from previous 1d (OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R3/S3 from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC for today's levels
    prev_high = high_1d[:-1]
    prev_low = low_1d[:-1]
    prev_close = close_1d[:-1]
    
    hl_range = prev_high - prev_low
    r3_levels = prev_close + hl_range * 1.1 / 2
    s3_levels = prev_close - hl_range * 1.1 / 2
    
    # Create daily arrays (skip first day)
    r3_per_day = np.full(len(df_1d), np.nan)
    s3_per_day = np.full(len(df_1d), np.nan)
    r3_per_day[1:] = r3_levels
    s3_per_day[1:] = s3_levels
    
    # Align to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_per_day)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_per_day)
    
    # Volume spike detection (20-period average on 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_4h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > R3, above 4h EMA20, volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_20_4h_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # Short: price < S3, below 4h EMA20, volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_20_4h_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price < S3 or below 4h EMA20
            if (close[i] < s3_aligned[i] or 
                close[i] < ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price > R3 or above 4h EMA20
            if (close[i] > r3_aligned[i] or 
                close[i] > ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA20 trend filter and volume confirmation.
# Camarilla levels identify key support/resistance from previous day's price action.
# Breakout above R3 with volume suggests bullish momentum; breakdown below S3 suggests bearish.
# 4h EMA20 filter ensures we only trade in the direction of the 4h trend.
# Volume confirmation ensures institutional participation.
# Works in bull markets (buy breakouts above R3 in uptrend) and bear markets (sell breakdowns below S3 in downtrend).
# Position size 0.20 balances risk and keeps trade frequency manageable (~15-30 trades/year).