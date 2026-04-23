#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA(50) trend filter + volume confirmation (>2x 20-period average)
- Donchian channels provide clear breakout levels with institutional significance
- 1w EMA(50) ensures trades align with weekly trend to avoid counter-trend whipsaws
- Volume spike (>2x average) validates breakout with institutional participation
- Designed for 1d timeframe targeting 7-25 trades/year (30-100 over 4 years)
- Works in both bull and bear markets by trading with weekly trend from daily breakouts
- Uses discrete position sizing (0.25) to minimize fee churn
"""

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
    
    # Get 1d data for Donchian calculation and 1w data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    # Upper = rolling max(high, 20)
    # Lower = rolling min(low, 20)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (no shift needed as we're already on 1d)
    upper_20_aligned = upper_20  # Already on 1d timeframe
    lower_20_aligned = lower_20  # Already on 1d timeframe
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        # Long: price breaks above upper Donchian level
        # Short: price breaks below lower Donchian level
        long_breakout = close[i] > upper_20_aligned[i]
        short_breakout = close[i] < lower_20_aligned[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long conditions: breakout above upper, uptrend, volume spike
            long_signal = (long_breakout and 
                          uptrend and
                          volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: breakout below lower, downtrend, volume spike
            short_signal = (short_breakout and 
                           downtrend and
                           volume[i] > 2.0 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Donchian level or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below lower Donchian level or trend turns down
                if (close[i] < lower_20_aligned[i] or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above upper Donchian level or trend turns up
                if (close[i] > upper_20_aligned[i] or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0