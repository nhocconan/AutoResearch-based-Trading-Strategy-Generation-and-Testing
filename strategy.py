#!/usr/bin/env python3
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
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend
    close_4h_series = pd.Series(close_4h)
    ema34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    volume_ma10_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma10_1d)
    
    # Calculate 1h ATR for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(volume_ma10_1d_aligned[i]) or
            np.isnan(atr[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 10-day average volume (aligned)
        volume_filter = volume[i] > (1.5 * volume_ma10_1d_aligned[i])
        
        # Volatility filter: ATR > 0.5 * 20-period ATR average (avoid low volatility chop)
        atr_ma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
        volatility_filter = atr[i] > (0.5 * atr_ma20[i])
        
        if position == 0:
            # Long: price above EMA34 with volume and volatility
            if close[i] > ema34_4h_aligned[i] and volume_filter and volatility_filter:
                signals[i] = 0.20
                position = 1
            # Short: price below EMA34 with volume and volatility
            elif close[i] < ema34_4h_aligned[i] and volume_filter and volatility_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA34
            if close[i] < ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above EMA34
            if close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA34_Trend_Volume_VolatilityFilter_Session"
timeframe = "1h"
leverage = 1.0