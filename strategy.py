#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
    # Long when price > upper Donchian + price > 1w EMA50 + volume > 2x 20-period average
    # Short when price < lower Donchian + price < 1w EMA50 + volume > 2x 20-period average
    # Exit when price crosses middle Donchian
    # Discrete position sizing: 0.25 to limit drawdown and reduce fee churn
    # Target: 30-100 total trades over 4 years (~7-25/year) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle = (upper + lower) / 2
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
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
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1w volume > 2 * 20-period average (strong volume expansion)
        vol_1w_current = df_1w['volume'].values
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w_current)
        volume_expansion = vol_1w_aligned[i] > 2.0 * vol_ma_aligned[i]
        
        # Breakout conditions with 1w EMA50 trend filter
        bullish_breakout = (close[i] > upper[i] and 
                           close[i] > ema_50_1w_aligned[i] and 
                           volume_expansion)
        bearish_breakout = (close[i] < lower[i] and 
                           close[i] < ema_50_1w_aligned[i] and 
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

name = "1d_1w_donchian_breakout_ema_volume_v1"
timeframe = "1d"
leverage = 1.0