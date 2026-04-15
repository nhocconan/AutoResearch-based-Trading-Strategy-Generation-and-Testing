#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h with 4h/1d regime filters and volume confirmation
# Use 4h EMA200 for long-term trend, 1d VWAP for institutional reference
# Enter long when price > 4h EMA200 AND price > 1d VWAP AND volume > 1.5x average
# Enter short when price < 4h EMA200 AND price < 1d VWAP AND volume > 1.5x average
# Exit when price crosses back through 4h EMA200
# Session filter: 08-20 UTC to avoid low-volume periods
# Target: 15-30 trades/year with 0.20 position size

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data once
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 4h EMA200 for trend filter
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # 1d VWAP approximation (typical price * volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = np.cumsum(typical_price_1d * volume_1d) / np.cumsum(volume_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_4h_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            continue
            
        # Check session and volume filters
        if not (session_filter[i] and vol_filter[i]):
            continue
            
        # Long conditions: price above both 4h EMA200 and 1d VWAP
        long_condition = (close[i] > ema200_4h_aligned[i]) and (close[i] > vwap_1d_aligned[i])
        
        # Short conditions: price below both 4h EMA200 and 1d VWAP
        short_condition = (close[i] < ema200_4h_aligned[i]) and (close[i] < vwap_1d_aligned[i])
        
        # Exit conditions: price crosses back through 4h EMA200
        exit_long = position == 1 and close[i] < ema200_4h_aligned[i]
        exit_short = position == -1 and close[i] > ema200_4h_aligned[i]
        
        if long_condition and position <= 0:
            position = 1
            signals[i] = position_size
        elif short_condition and position >= 0:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_4h1d_EMA200_VWAP_Volume"
timeframe = "1h"
leverage = 1.0