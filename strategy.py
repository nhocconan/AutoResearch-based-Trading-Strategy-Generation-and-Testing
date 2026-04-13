#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
    # Long when price > upper Donchian + price > 1d EMA50 + volume > 2x 20-period average
    # Short when price < lower Donchian + price < 1d EMA50 + volume > 2x 20-period average
    # Exit when price crosses middle Donchian
    # Discrete position sizing: 0.25 to limit drawdown and reduce fee churn
    # Target: 75-200 total trades over 4 years (~19-50/year) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle = (upper + lower) / 2
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume average (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 2 * 20-period average (strong volume expansion)
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        volume_expansion = vol_1d_aligned[i] > 2.0 * vol_ma_aligned[i]
        
        # Breakout conditions with 1d EMA50 trend filter
        bullish_breakout = (close[i] > upper[i] and 
                           close[i] > ema_50_1d_aligned[i] and 
                           volume_expansion)
        bearish_breakout = (close[i] < lower[i] and 
                           close[i] < ema_50_1d_aligned[i] and 
                           volume_expansion)
        
        # Exit condition: price returns to middle Donchian
        long_exit = close[i] < middle[i]
        short_exit = close[i] > middle[i]
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_ema_volume_v1"
timeframe = "4h"
leverage = 1.0