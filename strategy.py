#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with 1h volume confirmation and 4h EMA50 trend filter
# Long when price breaks above 1d Donchian(20) upper band AND 4h EMA50 is rising AND 1h volume > 1.5 * avg_volume(20)
# Short when price breaks below 1d Donchian(20) lower band AND 4h EMA50 is falling AND 1h volume > 1.5 * avg_volume(20)
# Exit when price returns to 1d Donchian(20) midpoint
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe
# 1d Donchian provides strong daily structure, filtering out 4h noise
# 4h EMA50 ensures we trade with the intermediate trend while reducing whipsaws
# 1h volume confirmation filters out low-conviction breakouts with higher timeframe validation
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "4h_1dDonchian20_Breakout_4hEMA50_Trend_1hVol"
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
    
    # Get 1d data ONCE before loop for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channel (20-period)
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    mid_1d = (upper_1d + lower_1d) / 2.0
    
    # Align 1d Donchian levels to 4h timeframe (wait for completed 1d bar)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    mid_1d_aligned = align_htf_to_ltf(prices, df_1d, mid_1d)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need at least 50 completed 4h bars for EMA50
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1h data ONCE before loop for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:  # Need at least 20 completed 1h bars for volume avg
        return np.zeros(n)
    volume_1h = df_1h['volume'].values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1h
    avg_volume_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1h = align_htf_to_ltf(prices, df_1h, volume_1h > (1.5 * avg_volume_20_1h))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(mid_1d_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_confirm_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper band, EMA50 rising, volume spike
            if (close[i] > upper_1d_aligned[i] and close[i-1] <= upper_1d_aligned[i-1] and 
                ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and volume_confirm_1h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower band, EMA50 falling, volume spike
            elif (close[i] < lower_1d_aligned[i] and close[i-1] >= lower_1d_aligned[i-1] and 
                  ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and volume_confirm_1h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1d Donchian midpoint
            if close[i] <= mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1d Donchian midpoint
            if close[i] >= mid_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals