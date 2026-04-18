#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Trend Continuation with 4h EMA filter and volume surge confirmation.
# In strong trends (price > 4h EMA), we look for pullbacks to resume the trend.
# Volume surge confirms institutional participation in the breakout.
# Works in bull markets (buy pullbacks in uptrends) and bear markets (sell rallies in downtrends).
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
name = "1h_EMA_Pullback_Volume_Surge"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA filter (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA (34-period) - trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1h EMA (21-period) for entry timing
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 20-period average volume for surge detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_21[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume surge: current volume 1.5x above average (institutional interest)
        volume_surge = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: uptrend (price > 4h EMA) + pullback to 1h EMA + volume surge
            uptrend = close[i] > ema_34_4h_aligned[i]
            pullback_to_ema = close[i] <= ema_21[i] * 1.02  # Allow small overshoot
            if uptrend and pullback_to_ema and volume_surge:
                signals[i] = 0.20
                position = 1
            # Short: downtrend (price < 4h EMA) + rally to 1h EMA + volume surge
            elif not uptrend and close[i] >= ema_21[i] * 0.98 and volume_surge:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend broken (price < 4h EMA) OR overextended (price > 1h EMA + 1.5%)
            trend_broken = close[i] < ema_34_4h_aligned[i]
            overextended = close[i] > ema_21[i] * 1.015
            if trend_broken or overextended:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend broken (price > 4h EMA) OR overextended (price < 1h EMA - 1.5%)
            trend_broken = close[i] > ema_34_4h_aligned[i]
            overextended = close[i] < ema_21[i] * 0.985
            if trend_broken or overextended:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals