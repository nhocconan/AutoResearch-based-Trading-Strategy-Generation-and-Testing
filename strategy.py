#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h EMA pullback with 4h/1d trend filter and volume confirmation
# Uses 4h EMA for trend direction, 1d EMA for long-term bias, and 1h EMA pullback for entry
# Volume confirmation reduces false signals. Session filter (08-20 UTC) avoids low-volume hours.
# Target: 15-37 trades/year (60-150 over 4 years) to minimize fee drag
name = "1h_ema_pullback_4h1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for long-term bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA(100) for long-term bias
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=100, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h EMA(20) for pullback entries
    ema_1h = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    # Calculate 1h ATR(14) for dynamic thresholds
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(ema_1h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            continue
        
        # Session filter: only trade during active hours (08-20 UTC)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Trend filters
        uptrend_4h = close[i] > ema_4h_aligned[i]
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_4h = close[i] < ema_4h_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: above average volume
        vol_confirm = volume[i] > vol_ma[i]
        
        # Dynamic entry threshold based on volatility
        entry_threshold = 0.001 * atr[i]  # 0.1% of ATR
        
        # Long entry: uptrend on both timeframes + pullback to EMA + volume
        if (uptrend_4h and uptrend_1d and 
            close[i] <= ema_1h[i] + entry_threshold and
            close[i] > ema_1h[i] - 2 * entry_threshold and  # Allow small pullback
            vol_confirm):
            signals[i] = 0.20
        
        # Short entry: downtrend on both timeframes + pullback to EMA + volume
        elif (downtrend_4h and downtrend_1d and 
              close[i] >= ema_1h[i] - entry_threshold and
              close[i] < ema_1h[i] + 2 * entry_threshold and  # Allow small pullback
              vol_confirm):
            signals[i] = -0.20
        
        # Otherwise maintain flat or previous signal (handled by carry-forward logic below)
        else:
            # Hold previous signal if we were in a trend, otherwise flat
            if i > 50:
                signals[i] = signals[i-1]
            else:
                signals[i] = 0.0
    
    return signals