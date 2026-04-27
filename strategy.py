#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion strategy using Bollinger Bands with volume confirmation and time-of-day filter.
# Goes long when price touches lower BB with volume spike and hour 8-20 UTC, short when touches upper BB with volume spike.
# Uses 4h trend filter to avoid counter-trend trades. Designed for 15-30 trades/year per symbol (60-120 total over 4 years).
# Works in both bull and bear markets by fading extremes in ranging conditions while respecting higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands on 1h close (20-period, 2 std)
    bb_length = 20
    bb_std = 2.0
    bb_ma = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    bb_upper = bb_ma + bb_std * bb_std_dev
    bb_lower = bb_ma - bb_std * bb_std_dev
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Get 4h data for trend filter (EMA 50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(bb_length, 20)  # 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Long conditions: price touches lower BB AND volume spike AND 4h uptrend (price > EMA50)
        if (low[i] <= bb_lower[i] and 
            volume_filter[i] and 
            close[i] > ema50_4h_aligned[i]):
            signals[i] = 0.20
            position = 1
        # Short conditions: price touches upper BB AND volume spike AND 4h downtrend (price < EMA50)
        elif (high[i] >= bb_upper[i] and 
              volume_filter[i] and 
              close[i] < ema50_4h_aligned[i]):
            signals[i] = -0.20
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_BBMeanReversion_Volume_4hTrendFilter_Session"
timeframe = "1h"
leverage = 1.0