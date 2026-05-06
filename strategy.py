#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian channel breakout with volume confirmation and 12h EMA trend filter
# Long when price breaks above 12h Donchian upper channel AND volume > 1.5 * avg_volume(20) AND 12h EMA50 is rising
# Short when price breaks below 12h Donchian lower channel AND volume > 1.5 * avg_volume(20) AND 12h EMA50 is falling
# Exit when price crosses the 12h Donchian midpoint (mean of upper and lower channel)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h Donchian provides stronger structural levels than shorter timeframes, reducing false breakouts
# 12h EMA50 ensures we trade with the intermediate trend while reducing noise vs shorter EMAs
# Volume confirmation (1.5x) filters weak breakouts and captures genuine institutional participation
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets by trading with the 12h trend

name = "4h_12hDonchian20_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian and EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need at least 50 completed 12h bars for EMA50
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channel (20-period)
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    midpoint_12h = (upper_12h + lower_12h) / 2.0
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h timeframe (wait for completed 12h bar)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    midpoint_aligned = align_htf_to_ltf(prices, df_12h, midpoint_12h)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    # Using 4h volume data directly since we're on 4h timeframe
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper channel, EMA50 rising, volume spike
            if (close[i] > upper_aligned[i] and 
                ema_50_aligned[i] > ema_50_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower channel, EMA50 falling, volume spike
            elif (close[i] < lower_aligned[i] and 
                  ema_50_aligned[i] < ema_50_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below the 12h Donchian midpoint
            if close[i] < midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above the 12h Donchian midpoint
            if close[i] > midpoint_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals