#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1w trend filter and volume confirmation
    # Long when price > upper Donchian + price above 1w EMA20 + volume > 1.5x 20-period average
    # Short when price < lower Donchian + price below 1w EMA20 + volume > 1.5x 20-period average
    # Exit when price crosses middle Donchian
    # Discrete position sizing: 0.25 to limit drawdown and reduce fee churn
    # Target: 50-150 total trades over 4 years (~12-38/year) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = high
    low_12h = low
    close_12h = close
    
    # Upper channel: highest high of last 20 periods
    upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Middle channel: average of upper and lower
    middle = (upper + lower) / 2
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1w volume average (20-period) with min_periods
    volume_1w = df_1w['volume'].values
    volume_series = pd.Series(volume_1w)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1w volume > 1.5 * 20-period average (volume expansion)
        vol_1w_current = df_1w['volume'].values
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w_current)
        volume_expansion = vol_1w_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions with 1w EMA trend filter
        bullish_breakout = (close[i] > upper[i] and 
                           close[i] > ema_20_1w_aligned[i] and 
                           volume_expansion)
        bearish_breakout = (close[i] < lower[i] and 
                           close[i] < ema_20_1w_aligned[i] and 
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

name = "12h_1w_donchian_breakout_ema_volume_v1"
timeframe = "12h"
leverage = 1.0